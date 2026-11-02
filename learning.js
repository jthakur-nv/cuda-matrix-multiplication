'use strict';
// Chapter explanations and exercises for the matrix multiplication tutorial.
const deepLessons = [
  {
    "heading": "Read the loops in the order they execute",
    "paragraphs": [
      "Suppose we want C[0,1]. The row loop chooses 0 and the column loop chooses 1. Keep those two numbers fixed while k moves through the inputs. The first step adds A[0,0] × B[0,1], the next adds A[0,1] × B[1,1], and so on. That is all the innermost loop does.",
      "The local variable sum starts at zero for each output. Once k has visited every position, sum holds the complete answer for that cell. Store it in C, then start another sum for the next column. Notice that calculating one cell does not require the answer from any other cell. We will use that independence when we reach CUDA.",
      "For an N×N result there are N² cells, with N multiply-and-add steps per cell. The usual operation count is 2N³. At N=2048 that is about 17.18 billion floating-point operations. All the implementations in this tutorial keep this same O(N³) algorithm; the changes concern how those operations are executed and supplied with data.",
      "Now follow the array indices. A[row*n+k] advances one float when k increases. B[k*n+col] advances n floats. At N=2048 that second step jumps 8 KiB. The values next to the requested B element may arrive in a cache line, but this loop does not use them immediately. That gives us a concrete place to start improving the code."
    ],
    "invariant": "Stop the inner loop after k products. The local sum should contain exactly those k contributions, with none missing or counted twice."
  },
  {
    "heading": "Why moving one loop changes memory access",
    "paragraphs": [
      "Take one value, A[row,k]. It contributes to every column in the same output row. The reordered loop loads that value into av, then walks along B[k,col] and C[row,col]. When col increases, both addresses advance by one float.",
      "There is one important consequence: we no longer finish a C cell in one uninterrupted dot product. We add one contribution to several cells, then return to them for the next k. This is why std::fill appears before the loops. Without that zeroing step, the first += would use an old or uninitialized value.",
      "A cache line contains a group of neighboring bytes. A loop that reads consecutive B values can use several values from a fetched line. The regular access pattern can also help prefetching and may let the compiler generate SIMD instructions, which operate on several columns at once. Inspect the generated code before assuming a particular SIMD instruction was used.",
      "Reordering does have a cost: the code updates C repeatedly rather than storing one final scalar sum. Whether that hurts depends on how those updates are handled by caches and registers. In the reference benchmark, the reordered version is faster. The CPU timing note explains why the exact speedup should be interpreted carefully."
    ],
    "invariant": "After processing the first k positions of a row, each C cell in that row should contain its first k contributions."
  },
  {
    "heading": "Choose a tile small enough to reuse",
    "paragraphs": [
      "Reordering fixes the direction of the inner memory access, but the matrices can still be much larger than a CPU cache. Blocking narrows the part of the problem being worked on. The ii and jj loops choose an output region; kk chooses which section of each dot product to add next.",
      "Inside those bounds, the i, k, j loops perform the same updates as the reordered version. Work through one output tile before looking at the whole nest. C does not get cleared when kk changes: the next K tile must add to the answer already accumulated there.",
      "With T=32, a rough footprint for three tiles is 3 × 32² × 4 bytes, or 12 KiB. This gives some intuition about the working set. It does not reserve 12 KiB of cache or guarantee that the tiles stay there. Cache organization, alignment, other data, and the compiler all affect what is retained.",
      "The min calls are worth checking. If N=5 and T=2, the last row tile starts at row 4 and has only one valid row. The clamped loop bounds skip row 5. Finally, check the timings: this blocked version is slower than the reordered version in the reference benchmark. A smaller working set is useful only if its benefit outweighs the extra loop work and changes to the generated code."
    ],
    "invariant": "For any valid C[i,j], the kk tiles must cover every k position exactly once. Initialize C before all the tiles, not inside the kk loop."
  },
  {
    "heading": "Work out the indices before writing the kernel",
    "paragraphs": [
      "The first CUDA version follows a simple rule: one thread owns one output cell. A 16×16 block covers 16 rows and 16 columns. The global column is blockIdx.x*16 + threadIdx.x; the row uses the same calculation with y. Try a few coordinates by hand before following the arithmetic loop.",
      "At N=2048, the grid has 128×128 blocks and 4,194,304 logical threads. The GPU schedules those blocks over time on its SMs. The grid is a description of all the work, rather than a count of physical processing units that must exist at once.",
      "Within a block, x changes fastest. In a 16×16 block, the first warp contains two rows of 16 threads. At a fixed k, each row of threads reads neighboring B elements and a common A element. This is the right way to examine coalescing: look at the addresses requested by neighboring lanes at the same instruction. Following one thread down B over the entire loop answers a different question.",
      "The host allocates GPU buffers, copies the inputs, and launches the kernel with <<<grid, block>>>. Launches are asynchronous, so the CPU can continue before the GPU finishes. The example checks launch errors and waits when validating results. Its CUDA-event timings cover GPU work with inputs already on the device; copies and allocation need separate timing if they matter to an application."
    ],
    "invariant": "Every valid row and column must map to exactly one thread. Only that thread writes the output cell. Out-of-range threads can return in this kernel because no block barrier follows."
  },
  {
    "heading": "Share the loads, then reuse the values",
    "paragraphs": [
      "The arithmetic is still an ordinary row-by-column dot product. What changes is how threads obtain its inputs. Neighboring outputs need many of the same A and B values. The walkthrough above shows four threads fetching those values together, then reading one another’s loads.",
      "For the small T=2 example, one phase performs eight multiply-and-add steps. The naïve kernel requests sixteen global input values; the shared-memory kernel loads eight and reuses each twice. For T=16, the equivalent counts are 8,192 and 512 global input loads per phase. These are requested scalar loads, not measured DRAM transactions: caches can already satisfy some naïve requests, and barriers also cost time.",
      "Consider a block computing a T×T output tile. For one K phase it needs T×T values from A and T×T from B. Loading these into shared memory gives the whole block a common copy. Those 2T² input values support T³ multiply-and-add steps, so each loaded value is used for T outputs.",
      "The first barrier handles a straightforward race. A fast thread might finish its load and start computing while another thread has not loaded its part of the tile. __syncthreads() makes the block wait until the shared inputs are ready.",
      "The second barrier matters just as much. After computing with a tile, a fast thread could begin loading the next one and overwrite a value that a slower thread is still reading. The second barrier makes the block finish reading before shared memory is reused. Each block has its own shared arrays; these barriers do not synchronize the whole grid.",
      "At an edge, an output thread may be outside the matrix but still have a job in loading the shared tile. Keep it in the barrier protocol. Load zero for an invalid input coordinate, and guard the final output store. For example, with N=20 and T=16 the final K tile has four valid positions, 16 through 19; the remaining positions become zero.",
      "At N=2048, a width of 16 needs 128 K phases; a width of 32 needs 64. That halves the number of barrier phases per block, but the larger block also uses more threads and shared memory. The size sweep lets us measure that trade rather than choosing a tile width from reuse or occupancy alone."
    ],
    "invariant": "Before computing, both shared arrays must hold the current K tile. Before replacing it, all readers must be finished. Invalid input coordinates contribute zero."
  },
  {
    "heading": "Keep the partial answers in registers",
    "paragraphs": [
      "The shared-memory kernel still loads two operands for each output contribution. We can reuse those operands further. Take four A values and four B values into a thread’s private variables. Their 4×4 outer product gives sixteen contributions to sixteen different outputs.",
      "The thread keeps those partial answers in acc[4][4]. For each k, eight shared values support sixteen FMAs. Computing those sixteen contributions separately would conceptually require thirty-two shared operand reads. The exercise lets you select an accumulator and follow its input values as k changes.",
      "The output positions are spaced apart. In the first block, thread (tx=3, ty=2) owns rows 2,18,34,50 and columns 3,19,35,51. Neighboring x threads still write neighboring columns within each group. A 256-thread block with sixteen outputs per thread covers 4,096 outputs: the full 64×64 tile.",
      "The shared inputs are a 64×16 A tile and a 16×64 B tile, totaling 8 KiB. Threads load several entries each using tid, tid+256, and so on. At N=2048 there are 128 K phases. Each accumulator keeps adding contributions until all phases are complete.",
      "Keeping more values live uses more registers. This build uses 41 registers per thread; the sixteen accumulators are only part of that total. Spilling would move some private values to memory and add accesses. Check the compiler report and spill counters rather than assuming an array stayed in registers. Also check the grid size: this kernel has only sixteen blocks at N=256, but 1,024 at N=2048."
    ],
    "invariant": "Each accumulator belongs to one output coordinate for the whole calculation. Its final store must use that same row and column, with a bounds check."
  },
  {
    "heading": "Check the layout at the library call",
    "paragraphs": [
      "SGEMM computes C=alpha*A*B+beta*C. Setting alpha=1 and beta=0 gives the operation used throughout this tutorial. Create the cuBLAS handle once and reuse it; initialization belongs outside the timed multiplication. The RAII wrapper in the full source releases the handle when it goes out of scope.",
      "The classic cuBLAS interface reads matrices as column-major, while the examples store rows consecutively. Interpreting a square row-major buffer as column-major gives its transpose without moving any bytes. To obtain the desired row-major result, the library must therefore calculate Cᵀ=BᵀAᵀ. That is why the call passes d_b before d_a.",
      "Try the wrong order in the 2×2 exercise. It produces B*A in the row-major output view, and matrix multiplication generally does not commute. For rectangular matrices, also work through the dimensions and leading dimensions; swapping pointers alone is not a general recipe.",
      "Pedantic FP32 mode keeps the arithmetic choice explicit so that reduced-precision TF32 is not silently mixed into this comparison. FP32 still rounds, and libraries can accumulate terms in a different order. The reference benchmark’s maximum absolute cuBLAS difference is 0.0002, within the validation tolerance.",
      "cuBLAS is the fastest implementation in the reference benchmark. The call itself does not reveal which internal kernel or instruction sequence was selected. Use it as a performance reference, and inspect a profile when a claim depends on its internal behavior. Remember -lcublas when building: the header declares the functions, and the library supplies their implementations."
    ],
    "invariant": "The output buffer, read as row-major, must contain A*B. Check layout, operand order, dimensions, and math mode together."
  }
];
const fmt = n => Number(n).toLocaleString('en-US');
function miniMatrix(label,n,cell,extra='') {
 return `<div class="mini-matrix"><span class="mini-label">${label}</span><div class="mini-grid ${extra}" style="--cols:${n}">${Array.from({length:n*n},(_,i)=>cell(Math.floor(i/n),i%n)).join('')}</div></div>`;
}
function gridCell(text,active=false,other='') {return `<span class="mini-cell ${active?'lit':''} ${other}">${text}</span>`;}
function mountLearningLab(i) {
 const box=document.getElementById('learning-lab');
 box.className='learning-lab';
 box.innerHTML=`<div class="lab-header"><span class="eyebrow">LEARN BY CHANGING SOMETHING</span><span>Interactive model · not GPU execution</span></div><div id="lab-controls" class="lab-controls"></div><div id="lab-view" class="lab-view"></div><div id="lab-explain" class="lab-explain" aria-live="polite"></div>`;
 [dotLab,localityLab,blockingLab,threadLab,sharedLab,registerLab,blasLab][i]();
 const detail=deepLessons[i];
 document.getElementById('deep-explanation').innerHTML=`<div class="deep-reading"><span class="section-index">READ THE CODE MORE CLOSELY</span><h4>${detail.heading}</h4>${detail.paragraphs.map(p=>`<p>${p}</p>`).join('')}<div class="invariant"><b>Check that the answer is still correct</b><p>${detail.invariant}</p></div></div>`;
}

function dotLab() {
 let k=0;const av=[1,2,3,4],bv=[2,1,3,2];
 document.getElementById('lab-controls').innerHTML='<button id="dot-next" class="outline-button">Add next product →</button><button id="dot-reset" class="outline-button">Reset</button><span class="model-caption">Small example: N=4 · one C cell</span>';
 const draw=()=>{
  const sum=av.slice(0,k).reduce((s,a,i)=>s+a*bv[i],0);
  document.getElementById('lab-view').innerHTML=`<div class="vector-pair"><div><span class="mini-label">A row</span><div class="vector">${av.map((v,i)=>gridCell(v,i<k)).join('')}</div></div><b>·</b><div><span class="mini-label">B column (shown sideways)</span><div class="vector">${bv.map((v,i)=>gridCell(v,i<k)).join('')}</div></div></div><div class="lab-stats"><div><span>TERMS INCLUDED</span><strong>${k} / 4</strong></div><div><span>LOCAL sum</span><strong>${sum}</strong></div><div><span>OUTPUT STATUS</span><strong>${k===4?'Ready to store':'Not complete'}</strong></div></div>`;
  document.getElementById('lab-explain').textContent=k===0?'sum starts at zero. Each click performs one more multiply-and-add from the inner k loop.':`${av.slice(0,k).map((v,i)=>`${v}×${bv[i]}`).join(' + ')} = ${sum}. ${k===4?'All four contributions are present; now C can receive the final sum.':'The highlighted entries have contributed; the remaining pairs still need to be added.'}`;
  document.getElementById('dot-next').disabled=k===4;
 };document.getElementById('dot-next').onclick=()=>{k++;draw();};document.getElementById('dot-reset').onclick=()=>{k=0;draw();};draw();
}
function localityLab() {
 document.getElementById('lab-controls').innerHTML='<label for="access-order">Innermost loop</label><select id="access-order"><option value="column">Basic: walk down B column</option><option value="row">Reordered: walk across B row</option></select>';
 const draw=()=>{
  const row=document.getElementById('access-order').value==='row',indices=Array.from({length:8},(_,k)=>row?8+k:k*8+1),lines=new Set(indices.map(x=>Math.floor(x/16)));
  document.getElementById('lab-view').innerHTML=miniMatrix('B · toy N=8 · numbers are flat indices',8,(r,c)=>gridCell(r*8+c,indices.includes(r*8+c)))+`<div class="address-trace"><span class="mini-label">ACCESSED ELEMENT INDICES</span><p>${indices.join(' → ')}</p><span class="mini-label">CACHE LINE NUMBERS (ILLUSTRATIVE)</span><p>${indices.map(x=>Math.floor(x/16)).join(' → ')}</p><strong>${lines.size} distinct cache line${lines.size===1?'':'s'}</strong></div>`;
  document.getElementById('lab-explain').textContent=`${row?'Each next B value is one float (4 bytes) away. One illustrative line contains this whole eight-value row.':'Each next B value is eight floats (32 bytes) away. The column touches four illustrative lines.'} This model assumes an aligned 64-byte cache line, 4-byte floats, and no eviction. It counts distinct lines for this short access sequence, not measured misses or runtime. Actual CPU cache behavior and SIMD require inspection.`;
 };document.getElementById('access-order').onchange=draw;draw();
}
function blockingLab() {
 let phase=0;
 document.getElementById('lab-controls').innerHTML='<label for="cpu-tile">Example tile width</label><select id="cpu-tile"><option>2</option><option selected>4</option><option>8</option></select><button id="cpu-phase" class="outline-button">Next K tile →</button>';
 const draw=()=>{
  const t=Number(document.getElementById('cpu-tile').value),base=phase*t;
  document.getElementById('lab-view').innerHTML=`<div class="three-matrices">${miniMatrix('A · first output row tile',8,(r,c)=>gridCell('',r<t&&c>=base&&c<base+t))}${miniMatrix('B · first output column tile',8,(r,c)=>gridCell('',r>=base&&r<base+t&&c<t))}${miniMatrix('C · accumulating into this tile',8,(r,c)=>gridCell('',r<t&&c<t))}</div><div class="lab-stats"><div><span>K TILE</span><strong>${phase+1} / ${8/t}</strong></div><div><span>THREE-TILE FOOTPRINT</span><strong>${3*t*t*4} bytes</strong></div><div><span>EXAMPLE 256-BYTE BUDGET</span><strong>${3*t*t*4<=256?'Fits by size':'Exceeds budget'}</strong></div></div>`;
  document.getElementById('lab-explain').textContent=`Small example: N=8: C rows 0…${t-1}, columns 0…${t-1} receive the k=${base}…${base+t-1} contributions. Changing K tiles keeps the same output region and adds another partial product. The footprint is 3×T²×4; the 256-byte budget is invented for teaching, not a measured CPU cache capacity. In the source, T=32 and the rough footprint is 12 KiB.`;
 };document.getElementById('cpu-tile').onchange=()=>{phase=0;draw();};document.getElementById('cpu-phase').onclick=()=>{phase=(phase+1)%(8/Number(document.getElementById('cpu-tile').value));draw();};draw();
}
function threadLab() {
 const n=20,t=16;
 document.getElementById('lab-controls').innerHTML='<span class="model-caption">Matrix 20×20 · blocks 16×16 · grid 2×2</span><label for="block-x">blockIdx.x (columns)</label><select id="block-x"><option>0</option><option>1</option></select><label for="block-y">blockIdx.y (rows)</label><select id="block-y"><option>0</option><option>1</option></select><label for="thread-id">Thread ID in block</label><input id="thread-id" type="range" min="0" max="255" value="35"><output for="thread-id" id="thread-value">35</output><label class="thread-padding-control"><input id="thread-padding" type="checkbox"> Show launched area, including padding</label>';
 const draw=()=>{
  const id=Number(document.getElementById('thread-id').value),bx=Number(document.getElementById('block-x').value),by=Number(document.getElementById('block-y').value),tx=id%t,ty=Math.floor(id/t),r=by*t+ty,c=bx*t+tx,valid=r<n&&c<n,warp=Math.floor(id/32),padding=document.getElementById('thread-padding').checked,extent=padding?32:20;
  const validRows=Math.min(t,n-by*t),validCols=Math.min(t,n-bx*t),validThreads=validRows*validCols;
  document.getElementById('thread-value').textContent=id;
  const axis=(value,extra='')=>`<span class="thread-axis ${extra}">${value}</span>`;
  let matrix=axis('r / c');for(let col=0;col<extent;col++)matrix+=axis(col);
  for(let row=0;row<extent;row++){
   matrix+=axis(row);
   for(let col=0;col<extent;col++){
    const inBlock=Math.floor(col/t)===bx&&Math.floor(row/t)===by,outside=row>=n||col>=n,selected=row===r&&col===c,inWarp=inBlock&&Math.floor(((row%t)*t+col%t)/32)===warp;
    const classes=['thread-cell',inBlock?'thread-block':'',inWarp?'thread-warp':'',outside?'thread-outside':'',selected?'thread-selected':'',col===16?'thread-block-left':'',row===16?'thread-block-top':'',col===20?'thread-matrix-right':'',row===20?'thread-matrix-bottom':''].join(' ');
    const tab=selected||(!valid&&!padding&&row===0&&col===0)?0:-1;
    matrix+=`<button class="${classes}" data-output-row="${row}" data-output-col="${col}" aria-label="${outside?'Padding':'Output C'}[${row},${col}], block (${Math.floor(col/t)},${Math.floor(row/t)}), thread (${col%t},${row%t})${outside?', outside 20 by 20 matrix':''}" aria-pressed="${selected}" tabindex="${tab}" title="${outside?'Outside matrix':'C'}[${row},${col}]">${outside?'×':''}</button>`;
   }
  }
  let local=axis('y / x');for(let x=0;x<t;x++)local+=axis(x);
  for(let y=0;y<t;y++){
   local+=axis(y);
   for(let x=0;x<t;x++){
    const outside=by*t+y>=n||bx*t+x>=n;
    local+=`<span class="thread-cell thread-block ${Math.floor((y*t+x)/32)===warp?'thread-warp':''} ${outside?'thread-outside':''} ${x===tx&&y===ty?'thread-selected':''}" title="threadIdx (${x},${y})${outside?' — no output':''}">${outside?'×':''}</span>`;
   }
  }
  document.getElementById('lab-view').innerHTML=`<div class="thread-map-layout"><div class="thread-map-main"><h4>Output matrix C: all 20 rows and 20 columns</h4><p class="caption">Numbers around the grid are global row and column indices. Click a cell to find its block and thread. The thick lines divide blocks.</p><div class="thread-matrix-scroll"><div class="thread-coordinate-grid" style="--extent:${extent}" role="group" aria-label="${padding?'20 by 20 output matrix within the 32 by 32 launched area':'Complete 20 by 20 output matrix'}">${matrix}</div></div><p class="caption">${padding?'The valid matrix ends at row 19 and column 19. Hatched cells are launched threads outside that boundary.':'Rows 0–15 and columns 0–15 belong to block index 0; rows 16–19 and columns 16–19 are the valid portion of block index 1.'}</p></div><div class="thread-map-detail"><h4>Block (${bx}, ${by}): all 256 threads</h4><p class="caption">Local x goes left to right; local y goes top to bottom. This view includes threads outside the output matrix.</p><div class="thread-matrix-scroll"><div class="thread-coordinate-grid thread-local-grid" style="--extent:16" aria-label="Local 16 by 16 thread block">${local}</div></div><p class="caption">${validThreads} threads write valid cells; ${256-validThreads} return without writing.</p><div class="address-trace"><span class="mini-label">SELECTED THREAD</span><p>threadIdx = (${tx}, ${ty})<br>warp = ${warp}, lane = ${id%32}<br>row = ${by}×16 + ${ty} = ${r}<br>col = ${bx}×16 + ${tx} = ${c}</p><strong>${valid?`Writes C[${r},${c}]`:'Outside matrix: return'}</strong><p class="caption">${valid?`Flat array index: ${r}×20 + ${c} = ${r*n+c}.`:'No C cell exists at this coordinate. Enable the padding view to locate the selected thread in the full launched area.'}</p></div></div></div><div class="thread-legend"><span><i class="thread-block"></i>Selected block</span><span><i class="thread-warp"></i>Same warp</span><span><i class="thread-selected"></i>Selected thread</span><span><i class="thread-outside"></i>Outside matrix</span></div><div class="thread-presets"><button class="outline-button" data-thread-example="0,0">Select C[0,0]</button><button class="outline-button" data-thread-example="19,19">Select C[19,19]</button><button class="outline-button" data-thread-example="20,20">Select outside [20,20]</button></div>`;
  document.getElementById('lab-explain').textContent=`A 20×20 matrix has valid indices 0 through 19. Each 16×16 block covers a full square, so the launch needs 2×2 blocks: 1,024 threads for 400 outputs. Block (${bx},${by}) covers rows ${by*t}–${by*t+15} and columns ${bx*t}–${bx*t+15}, with ${validThreads} valid output cells. ${valid?`The selected thread maps to C[${r},${c}]; it is highlighted in both views.`:`The selected thread maps to [${r},${c}], beyond the matrix boundary, so the naïve kernel returns.`} A warp contains 32 threads and spans two rows in this block. The early return applies to this naïve kernel; shared-memory kernels must keep their barrier protocol.`;
  const select=(row,col,focus=false)=>{
   document.getElementById('block-x').value=Math.floor(col/t);document.getElementById('block-y').value=Math.floor(row/t);document.getElementById('thread-id').value=(row%t)*t+col%t;
   draw();if(focus)document.querySelector(`[data-output-row="${row}"][data-output-col="${col}"]`)?.focus({preventScroll:true});
  };
  document.querySelectorAll('[data-output-row]').forEach(cell=>{
   cell.onclick=()=>select(Number(cell.dataset.outputRow),Number(cell.dataset.outputCol),true);
   cell.onkeydown=event=>{
    const movement={ArrowLeft:[0,-1],ArrowRight:[0,1],ArrowUp:[-1,0],ArrowDown:[1,0]}[event.key];
    if(!movement)return;event.preventDefault();
    select(Math.max(0,Math.min(extent-1,Number(cell.dataset.outputRow)+movement[0])),Math.max(0,Math.min(extent-1,Number(cell.dataset.outputCol)+movement[1])),true);
   };
  });
  document.querySelectorAll('[data-thread-example]').forEach(button=>button.onclick=()=>{const [row,col]=button.dataset.threadExample.split(',').map(Number);if(row>=n||col>=n)document.getElementById('thread-padding').checked=true;select(row,col);});
 };
 ['block-x','block-y','thread-id','thread-padding'].forEach(id=>document.getElementById(id).addEventListener('input',draw));draw();
}
function sharedLab() {
 document.getElementById('lab-view').innerHTML="<div id=\"cuda-tile-walk\" aria-label=\"Interactive shared memory matrix multiplication\">\n  <span class=\"sr-only\">Follow four threads loading shared tiles and computing a two by two output tile. All matrix values are illustrative.</span>\n  <h3>Four threads, one shared tile</h3>\n  <div class=\"model-caption\">Illustrative 4 × 4 matrices · T = 2 · block (x = 1, y = 0)</div>\n  <div class=\"shared-thread-control\">\n    <label class=\"shared-thread-label\">Follow a thread\n      <select class=\"shared-thread-select\" data-thread>\n        <option value=\"0\">(y=0, x=0) → C[0,2]</option>\n        <option value=\"1\">(y=0, x=1) → C[0,3]</option>\n        <option value=\"2\">(y=1, x=0) → C[1,2]</option>\n        <option value=\"3\" selected>(y=1, x=1) → C[1,3]</option>\n      </select>\n    </label>\n  </div>\n  <div class=\"walk-state\" aria-live=\"polite\">\n    <span data-stage></span><span data-position></span>\n  </div>\n  <div class=\"walk-pair\" role=\"group\" aria-label=\"Input matrices in global memory\">\n    <section><div class=\"walk-heading\">A · global memory</div><div data-a class=\"walk-matrix\" role=\"img\"></div></section>\n    <section><div class=\"walk-heading\">B · global memory</div><div data-b class=\"walk-matrix\" role=\"img\"></div></section>\n  </div>\n  <div class=\"walk-transfer\"><span data-transfer></span></div>\n  <div class=\"walk-pair walk-shared\" role=\"group\" aria-label=\"Shared memory visible to all four threads in this block\">\n    <section><div class=\"walk-heading\">as · shared by this block</div><div data-as class=\"walk-matrix\" role=\"img\"></div></section>\n    <section><div class=\"walk-heading\">bs · shared by this block</div><div data-bs class=\"walk-matrix\" role=\"img\"></div></section>\n  </div>\n  <div class=\"walk-key shared-small\"><span>Shaded: current tile</span><span>Outlined: followed thread’s loads</span><span>Filled: current operands</span></div>\n  <div class=\"walk-register\" aria-live=\"polite\"><span data-expression></span><span data-sum></span></div>\n  <div class=\"walk-code\"><code data-code></code></div>\n  <div class=\"walk-buttons\">\n    <button type=\"button\" class=\"outline-button\" data-back>Previous step</button>\n    <button type=\"button\" class=\"outline-button shared-next\" data-next>Next step</button>\n    <span data-step class=\"shared-small\"></span>\n  </div>\n</div>";

  const root=document.getElementById('cuda-tile-walk');
  const q=s=>root.querySelector(s);
  const A=Array.from({length:4},(_,r)=>Array.from({length:4},(_,c)=>r*4+c+1));
  const B=Array.from({length:4},(_,r)=>Array.from({length:4},(_,c)=>r*4+c+1));
  let step=0;
  const stages=['Load tiles','Wait for all loads','Multiply · k = 0','Multiply · k = 1','Finish reading tiles','Load next tiles','Wait for all loads','Multiply · k = 0','Multiply · k = 1','Finish reading tiles','Write the output'];
  function grid(el,n,values,cellClass,label){
    el.style.gridTemplateColumns=`20px repeat(${n}, minmax(0,1fr))`;
    el.setAttribute('aria-label',label);
    const nodes=[];
    for(let r=-1;r<n;r++)for(let c=-1;c<n;c++){
      const cell=document.createElement('span');
      if(r<0||c<0){cell.className='walk-axis';cell.textContent=r<0&&c<0?'':String(Math.max(r,c));}
      else{cell.className='walk-cell '+cellClass(r,c);cell.textContent=String(values(r,c));}
      nodes.push(cell);
    }
    el.replaceChildren(...nodes);
  }
  function render(){
    const id=Number(q('[data-thread]').value),y=Math.floor(id/2),x=id%2,row=y,col=2+x;
    const base=step<5?0:2,local=step<5?step:step-5;
    const computing=local===2||local===3;
    const k=local===3?1:0;
    let count=0;
    if(step<5)count=step>=3?2:step===2?1:0;
    else count=step>=8?4:step===7?3:2;
    let sum=0;
    for(let j=0;j<count;j++)sum+=A[row][j]*B[j][col];
    const aLoad=[row,base+x],bLoad=[base+y,col];
    grid(q('[data-a]'),4,(r,c)=>A[r][c],(r,c)=>[
      r<2&&c>=base&&c<base+2?'walk-tile':'',
      r===aLoad[0]&&c===aLoad[1]?'walk-own':'',
      computing&&r===row&&c===base+k?'walk-read':''
    ].join(' '),'Matrix A; rows and columns indexed zero through three');
    grid(q('[data-b]'),4,(r,c)=>B[r][c],(r,c)=>[
      r>=base&&r<base+2&&c>=2?'walk-tile':'',
      r===bLoad[0]&&c===bLoad[1]?'walk-own':'',
      computing&&r===base+k&&c===col?'walk-read':''
    ].join(' '),'Matrix B; rows and columns indexed zero through three');
    grid(q('[data-as]'),2,(r,c)=>A[r][base+c],(r,c)=>[
      'walk-tile',r===y&&c===x?'walk-own':'',computing&&r===y&&c===k?'walk-read':''
    ].join(' '),'Shared tile as; each cell was loaded by the thread at its local row and column');
    grid(q('[data-bs]'),2,(r,c)=>B[base+r][2+c],(r,c)=>[
      'walk-tile',r===y&&c===x?'walk-own':'',computing&&r===k&&c===x?'walk-read':''
    ].join(' '),'Shared tile bs; each cell was loaded by the thread at its local row and column');

    const explanation=document.getElementById('lab-explain');
    if(local===0) explanation.textContent='Each of the four threads loads one A value and one B value. Together they fill both shared tiles with eight global-memory loads. The dashed outlines mark the two values loaded by the followed thread. Every thread can read the entire shared tile after the first barrier.';
    else if(computing) explanation.textContent=`Thread (y=${y}, x=${x}) owns C[${row},${col}]. It now reads as[${y}][${k}] and bs[${k}][${x}], loaded by threads (y=${y}, x=${k}) and (y=${k}, x=${x}). Every A tile value is reused across two output columns, and every B tile value across two output rows. The arithmetic is an ordinary dot product; sharing these loads is the optimization.`;
    else if(step===10) explanation.textContent=`All four k positions have contributed. This thread writes its private sum, ${sum}, to C[${row},${col}]. The other three threads write their own output cells. Over both phases, this block requests 16 global input values instead of the naïve kernel’s 32; caches mean this is not necessarily a twofold reduction in DRAM traffic.`;
    else explanation.textContent=local===1?'The first __syncthreads() waits for all four threads to finish loading. A thread may need a value loaded by a different thread, so starting multiplication before that load finishes would be a race.':'The second __syncthreads() waits for all four threads to finish reading this tile. Only then may the next phase overwrite shared memory. Each thread keeps its private sum as base advances.';

    q('[data-stage]').textContent=stages[step];
    q('[data-position]').textContent=`base = ${base} · C[${row},${col}]`;
    q('[data-sum]').textContent=`Private sum = ${sum}`;
    q('[data-step]').textContent=`Step ${step+1} of 11`;
    q('[data-back]').disabled=step===0;
    q('[data-next]').disabled=step===10;
    if(local===0){
      q('[data-transfer]').textContent=`↓ A[${row},${base+x}] → as[${y},${x}] · B[${base+y},${col}] → bs[${y},${x}]`;
      q('[data-expression]').textContent='4 threads × 2 loads each';
      q('[data-code]').textContent='as[y][x] = A[row][base+x]; bs[y][x] = B[base+y][col];';
    }else if(computing){
      const av=A[row][base+k],bv=B[base+k][col],prev=sum-av*bv;
      q('[data-transfer]').textContent=`as[${y},${k}] × bs[${k},${x}] → private sum`;
      q('[data-expression]').textContent=`${prev} + ${av} × ${bv} = ${sum}`;
      q('[data-code]').textContent=`sum = fmaf(as[${y}][${k}], bs[${k}][${x}], sum);`;
    }else if(step===10){
      q('[data-transfer]').textContent=`Private register → C[${row},${col}]`;
      q('[data-expression]').textContent=`C[${row},${col}] = ${sum}`;
      q('[data-code]').textContent=`c[${row} * 4 + ${col}] = sum;`;
    }else{
      q('[data-transfer]').textContent='__syncthreads() · all four threads';
      q('[data-expression]').textContent=local===1?'All tile values ready':'All tile reads finished';
      q('[data-code]').textContent=local===1?'__syncthreads(); // ready to read shared tiles':'__syncthreads(); // safe to overwrite shared tiles';
    }
  }
  q('[data-back]').addEventListener('click',()=>{step=Math.max(0,step-1);render();});
  q('[data-next]').addEventListener('click',()=>{step=Math.min(10,step+1);render();});
  q('[data-thread]').addEventListener('change',render);
  render();

}
function registerLab() {
 let selected=0;
 document.getElementById('lab-controls').innerHTML='<label for="register-k">k within one K tile</label><input id="register-k" type="range" min="0" max="15" value="0"><output id="register-k-value" for="register-k">0</output><span class="model-caption">Click an accumulator</span>';
 const draw=()=>{
  const k=Number(document.getElementById('register-k').value),si=Math.floor(selected/4),sj=selected%4,ar=Array.from({length:4},(_,i)=>i+1+k),br=Array.from({length:4},(_,j)=>j+2+k),acc=(i,j)=>Array.from({length:k+1},(_,q)=>(i+1+q)*(j+2+q)).reduce((s,v)=>s+v,0);
  document.getElementById('register-k-value').textContent=k;
  document.getElementById('lab-view').innerHTML=`<div class="register-layout"><div><span class="mini-label">4 A VALUES IN REGISTERS</span><div class="vector">${ar.map((x,i)=>gridCell(x,i===si)).join('')}</div><span class="mini-label">4 B VALUES IN REGISTERS</span><div class="vector">${br.map((x,j)=>gridCell(x,j===sj)).join('')}</div><p class="caption">Illustrative values vary with k;<br>these are not the benchmark inputs.</p></div><div class="mini-matrix"><span class="mini-label">16 PRIVATE ACCUMULATORS · after k=${k}</span><div class="mini-grid" style="--cols:4">${Array.from({length:16},(_,q)=>`<button class="mini-cell ${q===selected?'chosen':''}" data-acc="${q}" aria-label="Accumulator ${Math.floor(q/4)}, ${q%4}" aria-pressed="${q===selected}">${acc(Math.floor(q/4),q%4)}</button>`).join('')}</div></div></div><div class="lab-stats"><div><span>SHARED VALUES READ PER k</span><strong>8 instead of 32</strong></div><div><span>FMAs PER THREAD PER k</span><strong>16</strong></div><div><span>SELECTED GLOBAL OUTPUT</span><strong>C[${2+si*16},${3+sj*16}]</strong></div></div>`;
  document.getElementById('lab-explain').textContent=`For the first block and thread (tx=3, ty=2), accumulator [${si}][${sj}] belongs to global row 2+${si}×16=${2+si*16}, column 3+${sj}×16=${3+sj*16}. At this k, ${ar[si]}×${br[sj]}=${ar[si]*br[sj]} is added to it; its sum over k=0…${k} is ${acc(si,sj)}. Selecting another accumulator shows how the same four A and four B values serve all sixteen outputs. This model shows one BK=16 phase; real outputs accumulate all 128 phases.`;
  document.querySelectorAll('[data-acc]').forEach(b=>b.onclick=()=>{selected=Number(b.dataset.acc);draw();document.querySelector(`[data-acc="${selected}"]`).focus({preventScroll:true});});
 };document.getElementById('register-k').oninput=draw;draw();
}
function blasLab() {
 document.getElementById('lab-controls').innerHTML='<label for="blas-order">Buffers passed to SGEMM</label><select id="blas-order"><option value="correct">B first, then A (our call)</option><option value="wrong">A first, then B (try the mistake)</option></select>';
 const draw=()=>{
  const correct=document.getElementById('blas-order').value==='correct',a=[[1,2],[3,4]],b=[[5,6],[7,8]],c=correct?[[19,22],[43,50]]:[[23,34],[31,46]];
  document.getElementById('lab-view').innerHTML=`<div class="three-matrices">${miniMatrix('A · row-major',2,(r,col)=>gridCell(a[r][col]))}${miniMatrix('B · row-major',2,(r,col)=>gridCell(b[r][col]))}${miniMatrix('Returned C · row-major view',2,(r,col)=>gridCell(c[r][col],correct))}</div><div class="address-trace"><span class="mini-label">WHAT THE COLUMN-MAJOR CALL SEES</span><p>${correct?'Bᵀ × Aᵀ = (A × B)ᵀ':'Aᵀ × Bᵀ = (B × A)ᵀ'}</p><strong>${correct?'Correct: row-major output is A × B':'Wrong target: row-major output is B × A'}</strong></div>`;
  document.getElementById('lab-explain').textContent=`A’s bytes contain [1,2,3,4]. A column-major view sees columns [1,2] and [3,4], giving Aᵀ without any copy. ${correct?'Passing B before A forms the transpose of the desired product. Interpreting the returned bytes as row-major gives [[19,22],[43,50]].':'Passing A before B with this unchanged call forms the transpose of B×A. Matrix multiplication is generally not commutative: B×A need not equal A×B.'} This is a 2×2 layout demonstration, not a live cuBLAS call or a performance model.`;
 };document.getElementById('blas-order').onchange=draw;draw();
}
window.mountLearningLab=mountLearningLab;
mountLearningLab(currentChapter);

// The resource capacities come from the supplied executable; the queued-work
// interaction is a simplified, single-SM capacity illustration.
document.getElementById('occupancy-lab').className='learning-lab';
document.getElementById('occupancy-lab').innerHTML=`<div class="lab-header"><span class="eyebrow">HOW CAN 24 BLOCKS FIT?</span><span>One-SM capacity model</span></div><div class="lab-controls"><label for="occ-kernel">Kernel</label><select id="occ-kernel">${resources.map(r=>`<option ${r[0]==='gpu_shared8'?'selected':''}>${r[0]}</option>`).join('')}</select><label for="queued-blocks">Blocks supplied to this SM</label><input id="queued-blocks" type="range" min="1" max="30" value="24"><output id="queued-value" for="queued-blocks">24</output></div><div id="occ-view" class="lab-view"></div><div id="occ-explain" class="lab-explain" aria-live="polite"></div>`;
function drawOccupancy(){
 const r=resources.find(r=>r[0]===document.getElementById('occ-kernel').value),queued=Number(document.getElementById('queued-blocks').value),resident=Math.min(queued,r[4]),warps=resident*r[1]/32;
 document.getElementById('queued-value').textContent=queued;
 document.getElementById('occ-view').innerHTML=`<div class="warp-model"><span class="mini-label">48 WARP SLOTS · each filled square is one resident warp in this model</span><div class="warp-slots">${Array.from({length:48},(_,i)=>`<span class="${i<warps?'occupied':''}" aria-label="Warp slot ${i+1}: ${i<warps?'occupied':'empty'}">${i+1}</span>`).join('')}</div></div><div class="lab-stats"><div><span>RESIDENT BLOCKS / CAPACITY</span><strong>${resident} / ${r[4]}</strong></div><div><span>ILLUSTRATED OCCUPANCY</span><strong>${(warps/48*100).toFixed(2)}%</strong></div><div><span>RESOURCE-BASED CEILING</span><strong>${r[5].toFixed(2)}%</strong></div></div>`;
 document.getElementById('occ-explain').textContent=`${r[1]} threads/block ÷ 32 = ${r[1]/32} warps/block. ${resident} resident blocks × ${r[1]/32} warps = ${warps} warps, or ${warps}/48 slots. ${Math.max(0,queued-resident)} supplied blocks wait in this model. Resident means present and eligible for scheduling, not all executing an instruction simultaneously. This is a capacity illustration using the compiled kernel’s resource limits, not measured achieved occupancy or the GPU’s real scheduling policy.`;
}
document.getElementById('occ-kernel').onchange=drawOccupancy;document.getElementById('queued-blocks').oninput=drawOccupancy;drawOccupancy();

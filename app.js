'use strict';
const results = [
  {id:'cpu_basic',name:'Basic CPU',device:'CPU',ms:8.9690,gflops:3.7412},
  {id:'cpu_reordered',name:'Reordered CPU',device:'CPU',ms:3.3653,gflops:9.9707},
  {id:'cpu_blocked',name:'Blocked CPU',device:'CPU',ms:4.1626,gflops:8.0609},
  {id:'gpu_naive',name:'Naïve CUDA',device:'GPU',ms:.0738,gflops:454.4405},
  {id:'gpu_shared8',name:'Shared 8 × 8',device:'GPU',ms:.0814,gflops:412.2409},
  {id:'gpu_shared16',name:'Shared 16 × 16',device:'GPU',ms:.8836,gflops:37.9754},
  {id:'gpu_shared32',name:'Shared 32 × 32',device:'GPU',ms:.0695,gflops:483.0812},
  {id:'gpu_register',name:'Register-tiled CUDA',device:'GPU',ms:.0832,gflops:403.0814},
  {id:'cublas',name:'cuBLAS',device:'GPU',ms:.0482,gflops:696.8208}
];
const lessons = [
{short:'Basic CPU',title:'Start with the definition.',tag:'CPU · ONE OUTPUT AT A TIME',result:'cpu_basic',
lead:'A direct translation of the math into three loops. Before thinking about speed, make the computation easy to understand and verify.',
analogy:'Imagine one person filling a spreadsheet. They finish the calculation for one cell, write it down, then move to the next cell.',
steps:['The outer loop picks a row of C. The next loop picks a column. Together they identify exactly one output cell.', 'Set a local sum to zero. The inner k loop multiplies matching entries from A’s row and B’s column, adding each product.', 'Store the completed sum into C. Each of the N² cells repeats the same N-step calculation.'],
mechanism:'A row of A is contiguous in memory. A column of B is not: advancing k changes the B index by N. At N=256, consecutive B values in this loop are 1,024 bytes apart because each float occupies 4 bytes. This access pattern can make poor use of CPU cache lines.',
code:`for (int row = 0; row < n; ++row)
    for (int col = 0; col < n; ++col) {
        float sum = 0;
        for (int k = 0; k < n; ++k)
            sum += a[size_t(row) * n + k]
                 * b[size_t(k) * n + col];
        c[size_t(row) * n + col] = sum;
    }`,
notes:[['Why size_t?','It uses an unsigned size type for array offsets. Casting before multiplication avoids a narrow integer multiplication in the index.'],['What is FP32?','A 32-bit floating-point number: C++ float. It represents a finite-precision approximation, not every real number exactly.'],['What stays private?','The sum belongs to one output calculation. Only the final result is stored in C.'],['What is our baseline?','The timed basic CPU implementation takes 8.9690 ms. The separate validation pass is excluded from this baseline.']],
tradeoff:'This is a single-threaded reference compiled with optimization. It is deliberately straightforward and is not representative of a tuned, multithreaded CPU matrix library.'},
{short:'Reorder the loops',title:'Walk along memory, not across it.',tag:'CPU · LOCALITY & REUSE',result:'cpu_reordered',
lead:'Keep the same multiplication, but change the loop order from row–column–k to row–k–column. Now the innermost loop walks across rows of both B and C.',
analogy:'Instead of repeatedly walking down different aisles for one item at a time, collect neighboring items in one pass while you are already there.',
steps:['Clear C to zero first. We will add partial contributions to each output rather than calculate one complete output in a local sum.', 'Pick a row and a k position. Load A[row,k] once into av.', 'Walk across every column, adding av × B[k,col] to the corresponding C element. The next k contributes another partial product to that row.'],
mechanism:'The inner loop reads consecutive B elements and updates consecutive C elements. A CPU cache line brings several nearby values together, and the loop provides an opportunity for SIMD: one instruction operating on multiple numbers. The compiler chooses whether and how to vectorize; the source alone does not prove it did so.',
code:`std::fill(c.begin(), c.end(), 0.0f);
for (int row = 0; row < n; ++row)
    for (int k = 0; k < n; ++k) {
        const float av = a[size_t(row) * n + k];
        for (int col = 0; col < n; ++col)
            c[size_t(row) * n + col] +=
                av * b[size_t(k) * n + col];
    }`,
notes:[['Why clear C?','The += operator reads the old value before adding to it. Without zero initialization, old results would corrupt the answer.'],['What is reused?','One A value serves all N columns, while a row of C is revisited for each k.'],['Why can it help?','Nearby accesses make cache lines more useful and make hardware prefetching easier.'],['What did you measure?','3.3653 ms, about 2.67× faster than basic CPU in this run. Zeroing C is included in this timing.']],
tradeoff:'The math is the same, but performance depends on caches, compiler decisions, and matrix size. Reordering is a useful first optimization because it changes relatively little code.'},
{short:'CPU cache blocking',title:'Work on a smaller patch.',tag:'CPU · CACHE BLOCKING',result:'cpu_blocked',
lead:'Wrap the reordered computation in loops over small 32 × 32 regions. These regions are called blocks or tiles. Here they are CPU loop regions, not CUDA thread blocks.',
analogy:'A desk holds only a few pages. Work on a small set of pages repeatedly before replacing them, rather than spreading an entire book across the desk.',
steps:['Initialize C to zero. The ii, kk, and jj loops choose which row, reduction, and column tiles to work on.', 'Within those regions, use the row–k–column order from the previous stage.', 'Use min(tile start + tile size, N) to stop at the matrix edge. The same code handles N that is not divisible by 32.'],
mechanism:'Blocking tries to reuse small portions of A, B, and C before the CPU evicts them from cache. The hardware manages CPU caches automatically; our loop order influences which values are accessed close together in time. It does not explicitly allocate shared memory.',
code:`constexpr int T = 32;
std::fill(c.begin(), c.end(), 0.0f);
for (int ii = 0; ii < n; ii += T)
  for (int kk = 0; kk < n; kk += T)
    for (int jj = 0; jj < n; jj += T)
      for (int i = ii; i < std::min(ii + T, n); ++i)
        for (int k = kk; k < std::min(kk + T, n); ++k) {
          const float av = a[size_t(i) * n + k];
          for (int j = jj; j < std::min(jj + T, n); ++j)
            c[size_t(i) * n + j] += av * b[size_t(k) * n + j];
        }`,
notes:[['Which tile size?','This version fixes T=32. A good CPU tile size depends on cache sizes, loop order, and compiler-generated code.'],['Does it change the answer?','It still adds each A[i,k] × B[k,j] contribution to C[i,j]. Bounds clamp partial tiles.'],['Where does reuse come from?','Repeated access to a limited working set gives cache-resident values a chance to serve multiple operations.'],['What did you measure?','4.1626 ms: faster than basic CPU, but slower than the reordered version. Extra loops can cost more than they save at small N.']],
tradeoff:'Blocking adds loop and boundary work. At N=256, the working set may already benefit from CPU caches, so blocking does not automatically win. Larger inputs and alternative tile sizes are worth measuring.'},
{short:'Naïve CUDA',title:'Give each output its own worker.',tag:'CUDA · THREAD HIERARCHY',result:'gpu_naive',
lead:'Move the arithmetic to the GPU. Thousands of independent output calculations become threads, grouped into blocks and launched together as a grid.',
analogy:'One person was filling the spreadsheet. Now a large team works in parallel, with each person responsible for one output cell.',
steps:['Launch blocks with dim3(16,16): 256 threads per block. Grid dimensions round N/16 upward so no output is missed.', 'Calculate the global row and column from blockIdx, blockDim, and threadIdx. At N=256 the grid has 16 × 16 = 256 blocks.', 'Threads within bounds run the entire k loop, keep one sum in a register, and store one output. Out-of-bounds threads can return because this kernel has no block barriers.'],
mechanism:'At a fixed k, adjacent x threads compute adjacent columns, so they read adjacent B elements. Threads in one output row read the same A element. Coalescing means combining neighboring lanes’ memory accesses into efficient transactions; it is a property of a group of threads at an instruction, not just one thread’s walk through its loop. Caches can also reuse repeated reads.',
code:`__global__ void gpu_naive(const float* a,
                          const float* b, float* c, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) return;
    float sum = 0;
    for (int k = 0; k < n; ++k)
        sum = fmaf(a[size_t(row) * n + k],
                   b[size_t(k) * n + col], sum);
    c[size_t(row) * n + col] = sum;
}
// Host launch: ceil-divide to cover every cell.
dim3 block(16, 16);
dim3 grid((n + 15) / 16, (n + 15) / 16);
gpu_naive<<<grid, block>>>(d_a, d_b, d_c, n);`,
notes:[['How do indices map?','A thread at x=3 in block x=2 produces column 2×16+3=35. The same formula maps y to the row.'],['What is fmaf?','A fused multiply-add computes a×b+sum with one final rounding. It can differ slightly from separate operations.'],['What is still repeated?','Many threads request the same input values. The kernel leaves reuse mostly to the hardware caches.'],['What did you measure?','0.0738 ms, about 121.53× the basic CPU compute speed. Host-device transfer time is excluded.']],
tradeoff:'Mapping independent work to threads delivers the first GPU gain. But launch overhead matters for small problems, and the baseline is not universally uncoalesced: distinguish repeated loads from strided lane accesses.'},
{short:'Shared memory tiling',title:'Let the team share its ingredients.',tag:'CUDA · COALESCED LOADS & SHARED MEMORY',result:'gpu_shared32',
lead:'Keep one thread per output, but have each block cooperatively load input tiles. Threads use these shared values repeatedly instead of requesting every operand separately from global memory.',
analogy:'A team brings a tray of ingredients to a shared workbench. Everyone uses the tray, then waits until the team is finished before replacing it.',
steps:['A T × T block owns a T × T output tile. In each K phase, every thread loads one A value and one B value into shared arrays.', 'Neighboring x threads load neighboring row-major elements. A barrier makes the completed tiles visible before computation.', 'Each thread performs T multiply-add steps from the shared tiles. A second barrier prevents early threads from overwriting values that others still need.', 'Advance to the next K tile. Partial loads become zero, but all threads still participate in both barriers. Store only outputs within bounds.'],
mechanism:'For a full K tile, the block loads 2T² input values and performs T³ multiply-add steps. Each input value serves T outputs in that block. Compare this with roughly 2T³ source-level operand reads without explicit tiling; real global traffic depends on caches. The shared arrays use 2 × T² × 4 bytes. At T=32 that is 8,192 bytes per block.',
code:`// Inside a T × T thread block:
__shared__ float as[T][T], bs[T][T];
float sum = 0;
for (int base = 0; base < n; base += T) {
    as[y][x] = (row < n && base+x < n)
        ? a[size_t(row)*n + base+x] : 0;
    bs[y][x] = (base+y < n && col < n)
        ? b[size_t(base+y)*n + col] : 0;
    __syncthreads();                 // tiles are ready
    #pragma unroll
    for (int k = 0; k < T; ++k)
        sum = fmaf(as[y][k], bs[k][x], sum);
    __syncthreads();                 // safe to replace tiles
}
if (row < n && col < n) c[size_t(row)*n + col] = sum;`,
notes:[['Why two barriers?','The first protects consumers from incomplete loads. The second protects remaining readers from the next phase’s writes.'],['Why no early return?','An invalid output thread can still need to load data for valid outputs. All participating block threads must reach the barriers consistently.'],['What does unroll do?','It asks the compiler to expand a fixed loop, potentially removing loop overhead. It is not an additional thread launch.'],['Why test three widths?','T=8,16,32 gives 64,256,1024 threads and 512,2048,8192 shared bytes per block. At N=256 there are 32,16,8 K phases, respectively.']],
tradeoff:'Larger tiles improve reuse and reduce phase count, but use more threads and shared memory per block. Shared32 took 0.0695 ms, shared8 took 0.0814 ms, and shared16 took 0.8836 ms. The shared16 result needs repeated measurement before a causal explanation.'},
{short:'Register tiling',title:'Give each worker a small patch.',tag:'CUDA · REGISTER REUSE',result:'gpu_register',
lead:'Instead of computing one output, each thread computes 16 outputs. A block still has 256 threads, but together they cover a larger 64 × 64 output region.',
analogy:'At the shared workbench, each worker keeps a few ingredients in their own hands and uses them for several products before reaching back to the tray.',
steps:['Launch a 16 × 16 block. It owns 64 × 64 outputs; A tiles are 64 × 16 and B tiles are 16 × 64.', 'Flatten threadIdx into tid. Threads take turns loading tile entries with idx=tid, tid+256, and so on. Linear loads group nearby global addresses.', 'For each k, load four A values and four B values from shared memory into private arrays. Form all 4 × 4 products, adding into 16 private accumulators.', 'Synchronize before replacing shared tiles. At the end, each thread stores its 16 valid output values.'],
mechanism:'For a single k, a thread reads 8 shared values and performs 16 multiply-adds, reusing each A and B fragment four times. A one-output approach would conceptually read 32 shared operands for those 16 products. The compiler can keep the small, unrolled accumulators in registers, although actual placement must be checked in the compiler resource report.',
code:`// Core computation from gpu_register:
float acc[4][4] = {};
// Cooperative loads and a barrier precede this loop.
#pragma unroll
for (int k = 0; k < 16; ++k) {
    float ar[4], br[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        ar[i] = as[ty + i*16][k];
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        br[j] = bs[k][tx + j*16];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
}
// For output (i,j) owned by this thread:
// row = blockIdx.y*64 + ty + i*16
// col = blockIdx.x*64 + tx + j*16`,
notes:[['A patch, not adjacent cells','A thread’s four rows and columns are each spaced by 16. This interleaving keeps neighboring x lanes’ output columns adjacent.'],['Shared footprint','64×16 + 16×64 = 2,048 floats, or 8,192 bytes per block. Equal shared bytes to shared32 does not mean equal work mapping.'],['Resource cost','Your build reports 41 registers per thread, versus 40 for shared kernels. The 16 accumulators are only part of a thread’s register use.'],['Parallelism at N=256','The grid is 4×4: only 16 blocks. At small N, too few blocks can leave SMs without work even when theoretical occupancy is high.']],
tradeoff:'Register tiling increases reuse but may increase register pressure and reduce grid parallelism. It took 0.0832 ms here, slower than naïve CUDA. This is an educational optimized kernel, not an architecture-tuned performance ceiling.'},
{short:'The cuBLAS reference',title:'Compare with a specialist.',tag:'CUDA LIBRARY · OPTIMIZED GEMM',result:'cublas',
lead:'Finish with NVIDIA’s matrix multiplication library. It supplies tuned implementations so we can compare our learning kernels with an established library reference.',
analogy:'After building your own tools, compare the job with a professional workshop. Its performance tells you there is more to learn, but the call does not reveal which internal technique ran.',
steps:['Create a cuBLAS handle once, outside the timed region. Set pedantic math mode to keep the comparison in FP32 rather than silently enabling TF32.', 'Call SGEMM: single-precision general matrix multiplication. alpha=1 and beta=0 mean C=AB with no contribution from old C.', 'Swap A and B in the call. cuBLAS interprets our row-major buffers as column-major transposes, and Cᵀ=BᵀAᵀ gives the desired row-major result.', 'Reuse the handle for timing and destroy it when finished. Link the program with -lcublas.'],
mechanism:'The library chooses its implementation; the console output does not identify its internal kernel or prove Tensor Core use. Here it is a reference for the same FP32 computation. Our learning kernels do not use Tensor Core instructions, asynchronous copy pipelines, or architecture-specific tuning.',
code:`cublasHandle_t handle;
cublasCreate(&handle);
cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
const float alpha = 1.0f, beta = 0.0f;

// Row-major C = A*B via column-major C^T = B^T*A^T.
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n, &alpha,
            d_b, n, d_a, n, &beta, d_c, n);

cublasDestroy(handle);
// Build: nvcc -O3 -std=c++17 matmul.cu -lcublas`,
notes:[['What is a handle?','A library context that carries settings and resources. Create it once rather than for every multiply.'],['What is a leading dimension?','The stride between columns in cuBLAS’s column-major view. For these square, contiguous matrices it is n.'],['Why pedantic FP32?','Changing arithmetic precision can change speed and accuracy. Keeping the math mode explicit makes this comparison clearer.'],['What did you measure?','0.0482 ms, about 186.08× faster than basic CPU compute and 1.53× faster than the naïve GPU kernel in this run.']],
tradeoff:'cuBLAS is the fastest recorded implementation here. Treat it as a tuned library comparison, not a guarantee for every size or proof that one particular internal optimization explains the result. The simplified snippet omits error checks; the downloadable source includes them.'}
];
const glossary = [
 ['Kernel','A function launched on the GPU, executed by many threads. __global__ marks a CUDA kernel callable from the host.'],
 ['Host / device','Host means the CPU side of the program; device means the GPU side. Data must be accessible where computation runs.'],
 ['SM','Streaming multiprocessor: a GPU processing unit that schedules warps and provides registers and shared memory.'],
 ['Warp','A group of 32 CUDA threads scheduled together. A 16×16 block’s first warp spans two rows of 16 threads.'],
 ['Coalescing','Combining nearby lanes’ global memory accesses into fewer efficient transactions. Alignment and the access pattern both matter.'],
 ['Tile','A small region of a matrix processed as a unit. It can describe a CPU cache region or a GPU shared/register working set.'],
 ['Barrier','A meeting point for cooperating threads. __syncthreads() waits for block threads and orders relevant memory accesses.'],
 ['Register pressure','The demand for registers per thread. More private values may limit resident work or cause spills to memory.'],
 ['Occupancy','Resident warps divided by the SM’s maximum warp capacity. Theoretical and achieved occupancy are different measurements.'],
 ['FP32 / TF32','FP32 is standard 32-bit float. TF32 is a Tensor Core arithmetic format with reduced significand precision for inputs; it is not enabled in this comparison.'],
 ['GFLOP/s','Billions of floating-point operations per second. Here it uses the conventional 2N³ count; it is not a memory bandwidth metric.'],
 ['Speedup','Baseline time divided by the new time. For example, 10 ms ÷ 2 ms = 5×. Always ask what the baseline and timing scope are.']
];
const escapeHtml = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
function highlight(code) {
  return code.split('\n').map(line => {
    const index = line.indexOf('//');
    const body = index >= 0 ? line.slice(0,index) : line;
    return escapeHtml(body).replace(/\b(for|if|return|const|int|float|void|constexpr|size_t|__global__|__shared__|__syncthreads)\b/g,'<span class="code-keyword">$1</span>')
      + (index >= 0 ? '<span class="code-comment">'+escapeHtml(line.slice(index))+'</span>' : '');
  }).join('\n');
}
let currentChapter = 0;
const tabs = document.getElementById('chapter-tabs');
const chapterNav = document.getElementById('chapter-nav');
lessons.forEach((lesson,i) => {
  const button = document.createElement('button');
  button.className = 'chapter-tab'; button.id='chapter-tab-'+i; button.textContent=String(i+1).padStart(2,'0');
  button.setAttribute('role','tab');button.setAttribute('aria-controls','lesson');button.setAttribute('aria-label',`Stage ${i+1}: ${lesson.short}`);
  button.addEventListener('click',()=>renderLesson(i));
  button.addEventListener('keydown',event=>{
    let next;
    if (event.key==='ArrowRight') next=(i+1)%lessons.length;
    if (event.key==='ArrowLeft') next=(i+lessons.length-1)%lessons.length;
    if (event.key==='Home') next=0;
    if (event.key==='End') next=lessons.length-1;
    if (next!==undefined) {event.preventDefault();renderLesson(next);tabs.children[next].focus();}
  });
  tabs.append(button);
  const link = document.createElement('button');link.className='chapter-link';
  link.innerHTML=`<span>${String(i+1).padStart(2,'0')}</span>${lesson.short}`;
  link.addEventListener('click',()=>{renderLesson(i);document.getElementById('lessons').scrollIntoView();});
  chapterNav.append(link);
});
function renderLesson(i) {
  currentChapter=i;
  const lesson=lessons[i], result=results.find(r=>r.id===lesson.result);
  [...tabs.children].forEach((tab,index)=>{tab.classList.toggle('selected',i===index);tab.setAttribute('aria-selected',i===index);tab.tabIndex=i===index?0:-1;});
  [...chapterNav.children].forEach((link,index)=>{link.classList.toggle('selected',i===index);link.setAttribute('aria-current',i===index?'step':'false');});
  const panel=document.getElementById('lesson');panel.setAttribute('aria-labelledby','chapter-tab-'+i);
  panel.innerHTML=`<div class="lesson-top"><div><span class="lesson-kicker">${lesson.tag}</span><h3>${lesson.title}</h3></div><div class="lesson-stat"><span>YOUR RUN · N=256</span><strong>${result.ms.toFixed(4)} <small>ms</small></strong><small>${result.name}</small></div></div>
  <p class="lesson-lead">${lesson.lead}</p><div class="analogy"><span class="tiny-label">THINK OF IT THIS WAY</span><p>${lesson.analogy}</p></div>
  <div class="lesson-body"><div><h4>Follow the work</h4><ol>${lesson.steps.map(step=>`<li>${step}</li>`).join('')}</ol></div><div><h4>What happens to the data?</h4><p>${lesson.mechanism}</p></div></div>
  <div class="terminal"><div class="code-top"><span>${i<3?'C++':'CUDA C++'} · ${i===5?'COMPUTATION EXCERPT':'LEARNING EXCERPT'}</span><button class="copy-button" id="copy-lesson">Copy code</button></div><pre><code>${highlight(lesson.code)}</code></pre></div>
  <p class="caption">Annotated learning excerpt. See the downloadable source for full declarations, allocation, launch setup, and error handling.</p>
  <div class="annotation-list">${lesson.notes.map(([title,text])=>`<p><b>${title}</b>${text}</p>`).join('')}</div><div class="tradeoff"><span class="tiny-label">THE TRADE-OFF</span><p>${lesson.tradeoff}</p></div>
  <div class="lesson-bottom"><button id="previous-lesson" ${i===0?'disabled':''}>← Previous idea</button><span>CHAPTER ${i+1} OF ${lessons.length}</span><button id="next-lesson">${i===lessons.length-1?'See the results ↓':'Next idea →'}</button></div>`;
  document.getElementById('copy-lesson').addEventListener('click',event=>copyText(lesson.code,event.currentTarget));
  document.getElementById('previous-lesson').addEventListener('click',()=>changeLesson(i-1));
  document.getElementById('next-lesson').addEventListener('click',()=>{if(i===lessons.length-1)document.getElementById('results').scrollIntoView();else changeLesson(i+1);});
}
function changeLesson(i) {renderLesson(i);document.getElementById('lessons').scrollIntoView();tabs.children[i].focus({preventScroll:true});}
async function copyText(text,button) {
  try {
    if(navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
    else {const area=document.createElement('textarea');area.value=text;area.style.position='fixed';area.style.opacity='0';document.body.append(area);area.select();const ok=document.execCommand('copy');area.remove();if(!ok)throw Error('Copy unavailable');}
    const original=button.textContent;button.textContent='Copied';setTimeout(()=>button.textContent=original,1600);
  } catch {button.textContent='Select text to copy';}
}
document.querySelectorAll('[data-copy]').forEach(button=>button.addEventListener('click',()=>copyText(document.getElementById(button.dataset.copy).textContent,button)));

const A=[[1,2,3],[4,5,6],[7,8,9]], B=[[2,0,1],[1,3,2],[0,1,4]];
const C=A.map(row=>B[0].map((_,col)=>row.reduce((sum,value,k)=>sum+value*B[k][col],0)));
let selectedRow=0, selectedCol=0;
function renderMatrix() {
  const demo=document.getElementById('matrix-demo');demo.innerHTML='';
  [['A',A],['B',B],['C',C]].forEach(([label,matrix],m)=>{
    if(m){const op=document.createElement('span');op.className='matrix-op';op.textContent=m===1?'×':'=';demo.append(op);}
    const wrapper=document.createElement('div');wrapper.innerHTML=`<div class="matrix-label">MATRIX ${label}</div>`;
    const grid=document.createElement('div');grid.className='matrix-grid';grid.setAttribute('aria-label','Matrix '+label);
    matrix.forEach((row,r)=>row.forEach((value,c)=>{
      const cell=document.createElement(m===2?'button':'span');
      cell.className='matrix-cell';cell.textContent=value;
      if((m===0&&r===selectedRow)||(m===1&&c===selectedCol))cell.classList.add('highlight');
      if(m===2){cell.setAttribute('aria-label',`C row ${r}, column ${c}: ${value}`);cell.setAttribute('aria-pressed',r===selectedRow&&c===selectedCol);if(r===selectedRow&&c===selectedCol)cell.classList.add('output');cell.addEventListener('click',()=>{selectedRow=r;selectedCol=c;renderMatrix();document.querySelectorAll('#matrix-demo button')[r*3+c].focus({preventScroll:true});});}
      grid.append(cell);
    }));wrapper.append(grid);demo.append(wrapper);
  });
  document.getElementById('dot-explanation').innerHTML=`C[${selectedRow}, ${selectedCol}] = ${A[selectedRow].map((v,k)=>`${v} × ${B[k][selectedCol]}`).join(' + ')} = <strong>${C[selectedRow][selectedCol]}</strong><small>Row ${selectedRow} of A meets column ${selectedCol} of B. Indices start at 0, just like in C++.</small>`;
}
let filter='all',sortDirection=0;
function renderResults() {
  const baseline=document.getElementById('baseline').value;
  const base=results.find(row=>row.id===baseline);
  const rows=results.filter(row=>filter==='all'||row.device===filter);
  if(sortDirection)rows.sort((a,b)=>sortDirection*(a.ms-b.ms));
  document.getElementById('speedup-heading').textContent='Speedup vs. '+(baseline==='cpu_basic'?'basic CPU':'naïve GPU');
  document.getElementById('result-rows').innerHTML=rows.map(row=>`<tr class="${row.id==='cublas'?'best':row.id==='gpu_shared16'?'flagged':''}"><td>${row.name}${row.id==='cublas'?'<span class="table-badge">FASTEST</span>':row.id==='gpu_shared16'?'<span class="table-badge flag">RECHECK</span>':''}<small>${row.id}</small></td><td>${row.device}</td><td>${row.ms.toFixed(4)}</td><td>${row.gflops.toFixed(4)}</td><td>${(base.ms/row.ms).toFixed(2)}×</td><td><span class="pass">PASS</span></td></tr>`).join('');
  document.getElementById('row-count').textContent=`Showing ${rows.length} of 9 measured implementations. ${sortDirection===0?'In learning order.':sortDirection===1?'Sorted by time: fastest first.':'Sorted by time: slowest first.'}`;
}
document.querySelectorAll('[data-filter]').forEach(button=>button.addEventListener('click',()=>{filter=button.dataset.filter;document.querySelectorAll('[data-filter]').forEach(b=>{b.classList.toggle('selected',b===button);b.setAttribute('aria-pressed',b===button);});renderResults();}));
document.getElementById('baseline').addEventListener('change',renderResults);
document.getElementById('sort-time').addEventListener('click',()=>{sortDirection=sortDirection===1?-1:1;document.getElementById('sort-time').closest('th').setAttribute('aria-sort',sortDirection===1?'ascending':'descending');renderResults();});
document.getElementById('export').addEventListener('click',()=>{
  const baseline=document.getElementById('baseline').value,base=results.find(r=>r.id===baseline);
  const csv=['implementation,device,N,iterations,time_ms,gflops,speedup_vs_'+baseline+',correctness',...results.map(r=>`${r.id},${r.device},256,5,${r.ms.toFixed(4)},${r.gflops.toFixed(4)},${(base.ms/r.ms).toFixed(4)},PASS`)].join('\r\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8;'})),a=document.createElement('a');a.href=url;a.download='matrix-lab-n256-results.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
const resources=[['gpu_naive',256,37,0,6,100],['gpu_shared8',64,40,512,24,100],['gpu_shared16',256,40,2048,6,100],['gpu_shared32',1024,40,8192,1,66.6667],['gpu_register',256,41,8192,5,83.3333]];
document.getElementById('occupancy-rows').innerHTML=resources.map(([name,threads,regs,shared,blocks,occupancy])=>`<tr><td>${name}</td><td>${threads.toLocaleString('en-US')}</td><td>${regs}</td><td>${shared.toLocaleString('en-US')}</td><td>${blocks}</td><td>${occupancy.toFixed(2)}%</td></tr>`).join('');
document.getElementById('glossary-grid').innerHTML=glossary.map(([term,meaning])=>`<div><h3>${term}</h3><p>${meaning}</p></div>`).join('');
if('IntersectionObserver' in window){const observer=new IntersectionObserver(entries=>{for(const entry of entries)if(entry.isIntersecting){document.querySelectorAll('[data-section]').forEach(link=>{const active=link.dataset.section===entry.target.id;link.classList.toggle('active',active);if(active)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');});}},{rootMargin:'-100px 0px -65% 0px',threshold:0});document.querySelectorAll('main>section').forEach(section=>observer.observe(section));}
renderLesson(0);renderMatrix();renderResults();

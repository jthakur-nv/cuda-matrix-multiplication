'use strict';
const profilerFigures=[
 {filename:'nsight-systems-timeline.png',src:'images/nsight-systems-timeline.png',title:'Nsight Systems: kernel launch intervals',alt:'Three Nsight Systems timeline plots compare ordinary launches and graph replay for shared16 at N=64, 256, and 2048.',caption:'Follow the filled segments from left to right: each is a kernel executing. The spaces between them become much less important as the matrix grows. Each row is aligned to its own first kernel; the gap percentages use all 20 kernels in the captured range.'},
 {filename:'nsight-compute-occupancy.png',src:'images/nsight-compute-occupancy.png',title:'Nsight Compute: theoretical and achieved occupancy',alt:'Grouped bar charts compare theoretical and achieved occupancy for five kernels at N=256 and N=2048.',caption:'Look at the register kernel in the two panels. Its resource limit stays at 83.33%, but achieved occupancy rises from 16.64% to 77.56% when the grid supplies more work. The measured occupancy uses active SM cycles, rather than whole-GPU utilization.'},
 {filename:'nsight-compute-memory.png',src:'images/nsight-compute-memory.png',title:'Nsight Compute: global and shared load activity',alt:'Horizontal bar charts show global-load requests and shared-load servicing wavefronts for five kernels at N=2048.',caption:'The register kernel is highlighted. It records 64× fewer global-load requests than naïve CUDA and about 4× fewer shared-load wavefronts than shared32. These L1/TEX-side counts describe servicing work; they are not DRAM-read counts.'},
 {filename:'nsight-compute-stalls.png',src:'images/nsight-compute-stalls.png',title:'Nsight Compute: barrier and memory-I/O stalls',alt:'Bar charts compare barrier and memory-I/O throttle stall ratios for five kernels at N=2048.',caption:'Compare the shared-memory kernels with the register kernel. Keeping input fragments in registers reduces repeated shared-memory work and changes the scheduler pressure. The vertical axes are stalled-warps-to-issuing-warps ratios, not percentages of elapsed time.'}
];
// Embedded image data also makes PNG downloads work when opened from disk.
for(const figure of profilerFigures) figure.src=window.profilerFigureData?.[figure.filename]??figure.src;
document.getElementById('hardware-counters').insertAdjacentHTML('afterend',`<div id="profiler-figures" class="profiler-figures"><h3>Profiler figures</h3><p>These images plot the saved Nsight exports. Choose a figure, then select the image to read it at full resolution. The source report and measurement settings are printed on each figure.</p><div class="lab-controls"><label for="profiler-figure">Figure</label><select id="profiler-figure">${profilerFigures.map((f,i)=>`<option value="${i}">${f.title}</option>`).join('')}</select></div><figure><button id="enlarge-profiler-figure" class="figure-preview" aria-label="Enlarge profiler figure"><img id="profiler-image" loading="lazy" alt=""></button><figcaption><p id="profiler-caption"></p><a id="download-profiler-image" class="text-link" download>Download PNG ↗</a></figcaption></figure></div><dialog id="profiler-image-dialog" class="figure-dialog" aria-labelledby="profiler-image-title"><div class="figure-dialog-toolbar"><h4 id="profiler-image-title"></h4><button id="fit-profiler-image" class="outline-button">Fit image</button><button id="close-profiler-image" class="outline-button">Close</button></div><p class="caption">Scroll to inspect the full-resolution image, or choose Fit image to see it all at once. Press Escape to close.</p><div class="figure-viewport"><img id="profiler-image-large" alt=""></div></dialog>`);
function updateProfilerFigure(){
 const figure=profilerFigures[Number(document.getElementById('profiler-figure').value)];
 const image=document.getElementById('profiler-image');image.src=figure.src;image.alt=figure.alt;
 document.getElementById('profiler-caption').textContent=figure.caption;
 const download=document.getElementById('download-profiler-image');download.href=figure.src;download.download=figure.filename;
}
document.getElementById('profiler-figure').onchange=updateProfilerFigure;
document.getElementById('enlarge-profiler-figure').onclick=()=>{
 const figure=profilerFigures[Number(document.getElementById('profiler-figure').value)],image=document.getElementById('profiler-image-large');
 image.src=figure.src;image.alt=figure.alt;image.classList.remove('fit-image');
 document.getElementById('profiler-image-title').textContent=figure.title;
 document.getElementById('fit-profiler-image').textContent='Fit image';
 document.getElementById('profiler-image-dialog').showModal();
};
document.getElementById('close-profiler-image').onclick=()=>document.getElementById('profiler-image-dialog').close();
document.getElementById('fit-profiler-image').onclick=event=>{const fit=document.getElementById('profiler-image-large').classList.toggle('fit-image');event.currentTarget.textContent=fit?'Full resolution':'Fit image';};
updateProfilerFigure();

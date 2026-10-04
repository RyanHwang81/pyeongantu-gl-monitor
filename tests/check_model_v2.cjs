const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('fs'),path=require('path'),url=require('url');
const out=process.env.GL_QA_OUTPUT||path.join(require('os').tmpdir(),'gl-model-v2-qa');fs.mkdirSync(out,{recursive:true});
const base=process.env.GL_QA_BASE;
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROME_EXECUTABLE?{executablePath:process.env.CHROME_EXECUTABLE}:{})});
 const cases=[];
 for(const lang of ['ko','en'])for(const width of [360,390,768,1024,1440]){
  const page=await browser.newPage({viewport:{width,height:1100}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
  const target=base?base+(lang==='en'?'en.html':'index.html')+'?qa='+Date.now():url.pathToFileURL(path.resolve(__dirname,'../dist',lang==='en'?'en.html':'index.html')).href;
  const response=await page.goto(target,{waitUntil:'domcontentloaded',timeout:60000});await page.locator('#recent-quad [data-month]').first().waitFor();
  const check=await page.evaluate(()=>({scroll:document.documentElement.scrollWidth,inner:innerWidth,lang:document.documentElement.lang,
    version:DATA.meta.model_version,latest,annual:ANN,months:M.length,regime:document.getElementById('kpi-regime').textContent,
    raw:document.getElementById('kpi-l-raw').textContent,bands:document.querySelectorAll('#recent-quad .boundary-band').length,
    selected:document.getElementById('status-sel').textContent,assets:document.querySelectorAll('#asset-rank .asset-row').length,
    method:document.getElementById('v1-comparison')!==null,body:document.body.innerText}));
  if(check.scroll>width+1||check.lang!==lang||check.version!=='2.0'||check.method||check.bands!==2||errors.length)throw Error(JSON.stringify({lang,width,scroll:check.scroll,method:check.method,bands:check.bands,errors}));
  if(check.latest.p&&(check.assets!==0||!check.body.includes(lang==='ko'?'잠정':'Provisional')))throw Error('provisional presentation');
  const expected=await page.evaluate(()=>annualize(M.filter(m=>!m.p)));
  if(JSON.stringify(check.annual)!==JSON.stringify(expected))throw Error('provisional annual average');
  const recent=page.locator('#recent-quad [data-month="'+check.latest.d+'"]');
  if(check.latest.p){
   if(await recent.getAttribute('data-provisional')!=='true')throw Error('provisional marker');
   const fill=await recent.locator('circle').evaluateAll(cs=>cs.some(c=>c.getAttribute('fill')==='#fffdf9'&&c.getAttribute('stroke')!=='#fffdf9'));
   if(!fill)throw Error('hollow month point');
  }
  await page.locator('#recent-period-sel').selectOption('m24');
  await page.locator('#recent-quad [data-month]').first().focus();await page.keyboard.press('Enter');
  if(await page.locator('#sel-date').textContent()===check.latest.d)throw Error('keyboard selection');
  await page.locator('#annual-period-sel').selectOption('d2000');
  const decade=await page.evaluate(()=>ANN.filter(a=>+a.d>=2000&&+a.d<2010).map(a=>a.d));
  const plotted=await page.locator('#annual-quad [data-month]').evaluateAll(nodes=>nodes.map(n=>n.getAttribute('data-month')));
  if(JSON.stringify(plotted)!==JSON.stringify(decade)||decade.length===0)throw Error('annual range '+JSON.stringify({lang,width,plotted,decade}));
  // A deterministic one-axis boundary fixture checks the display even when the
  // real latest month falls inside the two-axis neutral box instead.
  const boundary=await page.evaluate(()=>{state.selected={...latest,g:-.01,l:.30,n:false,b:true,p:false,c:false};updateSelPanel();return {label:document.getElementById('sel-regime').textContent,opacity:document.getElementById('sel-regime').style.opacity,duration:regimeDuration([{d:'a',r:'expansion'},{d:'b',r:'defense',b:true},{d:'c',r:'expansion'},{d:'d',r:'defense',p:true}])};});
  if(boundary.opacity!=='0.55'||boundary.duration.count!==3||!boundary.duration.boundary||!boundary.label.includes(lang==='ko'?'경계':'Boundary'))throw Error('boundary presentation');
  await page.evaluate(()=>{state.selected=latest;state.recentPreset='m12';state.annualPreset='all';document.getElementById('recent-period-sel').value='m12';document.getElementById('annual-period-sel').value='all';renderRecent();renderAnnual();updateSelPanel();drawRibbon();});
  await page.screenshot({path:path.join(out,(base?'public':'local')+'-'+lang+'-'+width+'.png'),fullPage:true});
  await page.locator('#quant-reference').screenshot({path:path.join(out,(base?'public':'local')+'-quant-'+lang+'-'+width+'.png')});
  if(width===1440)fs.writeFileSync(path.join(out,'visible-'+lang+'.txt'),check.body);
  cases.push({lang,width,months:check.months,latest:check.latest,regime:check.regime,bodyKorean:(check.body.match(/[가-힣]/g)||[]).length,status:response?.status()||200});await page.close();
 }
 const page=await browser.newPage();await page.goto(url.pathToFileURL(path.resolve(__dirname,'../dist/gl-internal.html')).href);
 if(await page.locator('#v1-comparison tbody tr').count()!==36)throw Error('internal v1 comparison');await page.close();
 const receipt={ok:true,cases,base:base||'local file',internalComparisonRows:36};fs.writeFileSync(path.join(out,(base?'public':'local')+'-receipt.json'),JSON.stringify(receipt,null,2));
 await browser.close();console.log(JSON.stringify({ok:true,cases:cases.length,englishKoreanCharacters:cases.find(x=>x.lang==='en').bodyKorean}));
})().catch(e=>{console.error(e);process.exit(1)});

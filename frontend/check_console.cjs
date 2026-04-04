const puppeteer = require('puppeteer');

(async () => {
  try {
    const browser = await puppeteer.launch({ headless: "new", args: ['--no-sandbox'] });
    const page = await browser.newPage();
    
    page.on('console', msg => console.log('PAGE LOG:', msg.text()));
    page.on('pageerror', err => console.log('PAGE ERROR:', err.toString()));
    page.on('requestfailed', request => console.log('REQUEST FAILED:', request.url(), request.failure().errorText));

    await page.goto('http://localhost:5173', { waitUntil: 'networkidle0', timeout: 10000 });
    
    const content = await page.content();
    console.log("Body length:", content.length);
    
    await browser.close();
  } catch (err) {
    console.error("Puppeteer Script Error:", err);
  }
})();

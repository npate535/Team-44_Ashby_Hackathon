// Wait until a remote command log ends with its exit marker, fetching it through the OnDemand Files app.
const P = JSON.parse(await fs.readFile('__PARAMS__', 'utf8'));
const url = `${P.ood}/pun/sys/dashboard/files/fs/${P.remote}?download=1`;
const deadline = Date.now() + P.waitMs;
let state = 'PQ_PENDING';
while (Date.now() < deadline) {
  const res = await fetch(url);
  const type = res.headers.get('content-type') || '';
  if (res.status === 200 && type.includes('text/html')) {
    // A missing file redirects to its directory listing; only a sign-in page means the session expired.
    if (!res.url.includes('/pun/sys/dashboard/files/')) {
      state = 'PQ_ERR session_expired';
      break;
    }
  } else if (res.status === 200) {
    const text = await res.text();
    await fs.writeFile(P.outFile, text);
    if (P.once || text.includes('__PQ_EXIT=')) {
      state = 'PQ_DONE';
      break;
    }
  }
  if (P.once) break;
  await sleep(2000);
}
console.log(state);

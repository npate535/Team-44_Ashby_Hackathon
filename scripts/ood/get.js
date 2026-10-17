// Download one remote file through the OnDemand Files app into the staging directory.
const P = JSON.parse(await fs.readFile('__PARAMS__', 'utf8'));
const res = await fetch(`${P.ood}/pun/sys/dashboard/files/fs/${P.remote}?download=1`);
const type = res.headers.get('content-type') || '';
if (res.status !== 200) {
  console.log(`PQ_ERR status ${res.status}`);
} else if (type.includes('text/html') && !P.allowHtml) {
  const reason = res.url.includes('/pun/sys/dashboard/files/') ? 'missing file or directory' : 'session expired';
  console.log(`PQ_ERR ${reason}`);
} else {
  const body = Buffer.from(await res.arrayBuffer());
  await fs.writeFile(P.outFile, body);
  console.log(`PQ_GOT ${body.length}`);
}

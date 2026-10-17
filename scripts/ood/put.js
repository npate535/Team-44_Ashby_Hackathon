// Upload one staged local file into a remote directory with the OnDemand Files app's upload dialog.
const P = JSON.parse(await fs.readFile('__PARAMS__', 'utf8'));
const local = `${pwd}/${P.name}`;
await fs.copyFile(P.stagedFile, local);
await openTab(`${P.ood}/pun/sys/dashboard/files/fs/${P.remoteDir}`);
const listing = await snapshot(page, { interactive: true });
const uploadButton = listing.tree.match(/button "Upload" \[ref=(e\d+)\]/);
if (!uploadButton) {
  console.log('PQ_ERR no_upload_button (session expired?)');
} else {
  await page.locator(uploadButton[1]).click();
  await sleep(600);
  await page.locator('input[type=file]').first().setInputFiles(local);
  await sleep(600);
  const dialog = await snapshot(page, { interactive: true });
  const confirm = dialog.tree.match(/button "Upload \d+ files?" \[ref=(e\d+)\]/);
  if (!confirm) {
    console.log('PQ_ERR no_confirm_button');
  } else {
    await page.locator(confirm[1]).click();
    const target = `${P.ood}/pun/sys/dashboard/files/fs/${P.remoteDir}/${P.name}?download=1`;
    let state = 'PQ_ERR upload_timeout';
    for (let i = 0; i < P.waitSeconds; i++) {
      await sleep(1000);
      const res = await fetch(target, { method: 'HEAD' });
      const size = Number(res.headers.get('content-length') || -1);
      if (res.status === 200 && size === P.size) {
        state = `PQ_PUT ${size}`;
        break;
      }
    }
    console.log(state);
  }
}

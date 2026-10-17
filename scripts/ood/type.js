// Type a prepared script into the persistent OnDemand "Delta Shell Access" tab.
const P = JSON.parse(await fs.readFile('__PARAMS__', 'utf8'));
// Prefer the pinned bridge tab so other browser work in a second shell tab never collides with ours.
const shellTabs = (await listBrowserTabs()).filter((t) => t.url.includes('/pun/sys/shell/ssh/'));
const pinned = shellTabs.find((t) => t.targetId === P.tabId) || shellTabs[0];
if (!pinned) {
  console.log('PQ_ERR no_shell_tab');
} else {
  console.log(`PQ_TAB ${pinned.targetId}`);
  await attachBrowserTab(pinned.targetId);
  const snap = await snapshot(page, { interactive: true });
  const term = snap.tree.match(/log [^\n]*\[ref=(f\d+e\d+)\]/);
  if (!term) {
    console.log('PQ_ERR no_terminal');
  } else {
    const lines = (await fs.readFile(P.typeFile, 'utf8')).split('\n');
    for (const line of lines) {
      if (line.length) await page.locator(term[1]).pressSequentially(line);
      await page.locator(term[1]).press('Enter');
    }
    console.log('PQ_TYPED');
  }
}

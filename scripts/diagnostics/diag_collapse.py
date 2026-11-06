import sys, numpy as np, torch, cv2
sys.path.insert(0, "/media/yunyi/Data/Desktop/work/vla/alicia-d-vla-benchmark/scripts")
sys.path.insert(0, "/media/yunyi/Data/Desktop/work/vla/TurboVLA/experiments/wipes")
from eval_alicia_turbovla import TurboVLAAliciaPolicy
from wipes_dataset import LeRobotWipesDataset
W="/media/yunyi/Data/Desktop/work/vla"
ds = LeRobotWipesDataset(dataset_dir=f"{W}/datasets/wipes-to-green-plate-01", stats_path=f"{W}/weights/turbovla_wipes/wipes_stats.json")
meta = ds.lerobot_ds.meta.episodes
eps = list(range(0, 71, 2))  # 36 episodes
items = []
for ep in eps:
    frm, to = int(meta[ep]["dataset_from_index"]), int(meta[ep]["dataset_to_index"])
    st = ds.all_states[frm:to]
    i0 = np.where(np.max(np.abs(st[:, :6] - st[0:1, :6]), axis=1) > 1.5)[0][0]
    g = frm + i0 + 40                       # approach phase, wipes location must be inferred from images
    it = ds.lerobot_ds[g]
    items.append(dict(top=it["observation.images.top"], wrist=it["observation.images.wrist"],
                      state=torch.tensor(ds.all_states[g]), gt=ds.all_actions[min(g+29, to-1)]))
def warp(t, s, rot, dx, dy):
    im = (t.permute(1,2,0).numpy()*255).astype(np.uint8); h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), rot, s); M[0,2] += dx; M[1,2] += dy
    out = cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return torch.from_numpy(out).permute(2,0,1).float()/255.
def run(pol, mode):
    P = []
    for it in items:
        top, wr, st = it["top"], it["wrist"], it["state"]
        if mode == "warp_top":  top = warp(top, 0.65, 15, 60, 20)
        if mode == "warp_top2": top = warp(top, 0.65, -15, -60, 20)
        if mode == "top_zero":  top = torch.zeros_like(top)
        if mode == "wrist_zero": wr = torch.zeros_like(wr)
        b = {"observation.images.top": top.unsqueeze(0).cuda(), "observation.images.wrist": wr.unsqueeze(0).cuda(), "observation.state": st.unsqueeze(0).cuda()}
        P.append(pol._predict_action_chunk(b)[0,29].cpu().numpy())
    return np.array(P)
GT = np.array([it["gt"] for it in items])
print(f"n={len(items)} | GT c[29] std across episodes: J1={GT[:,0].std():.1f} J2={GT[:,1].std():.1f} J3={GT[:,2].std():.1f}")
for ck in ["turbovla_wipes_freeze_vit_step_08000.pth", "turbovla_wipes_warm_full_step_12500.pth"]:
    pol = TurboVLAAliciaPolicy(ckpt_path=f"{W}/weights/turbovla_wipes/{ck}", stats_path=f"{W}/weights/turbovla_wipes/wipes_stats.json",
        dinov3_path=f"{W}/models/dinov3-vitb16-pretrain-lvd1689m", bert_path=f"{W}/models/bert-base-uncased", device="cuda")
    print(f"\n=== {ck} ===")
    for mode in ["clean", "warp_top", "warp_top2", "top_zero", "wrist_zero"]:
        P = run(pol, mode)
        e = np.abs(P[:, :3] - GT[:, :3]).mean(0)
        print(f"  {mode:10s} MAE(J1,J2,J3)=({e[0]:.1f},{e[1]:.1f},{e[2]:.1f}) | pred std across eps J1={P[:,0].std():.1f} J2={P[:,1].std():.1f} J3={P[:,2].std():.1f} | pred mean J3={P[:,2].mean():.1f} (GT mean {GT[:,2].mean():.1f})")

import sys, glob, av, numpy as np, cv2
sys.path.insert(0, "/media/yunyi/Data/Desktop/work/vla/TurboVLA/experiments/wipes")
from wipes_dataset import LeRobotWipesDataset
ds = LeRobotWipesDataset(dataset_dir="/media/yunyi/Data/Desktop/work/vla/datasets/wipes-to-green-plate-01",
    stats_path="/media/yunyi/Data/Desktop/work/vla/weights/turbovla_wipes/wipes_stats.json")
meta = ds.lerobot_ds.meta.episodes
def train_f0(ep):
    im = (ds.lerobot_ds[int(meta[ep]["dataset_from_index"])]["observation.images.top"].permute(1,2,0).numpy()*255).astype(np.uint8)
    return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
def dec(v, n=0):
    c = av.open(v)
    for i, fr in enumerate(c.decode(video=0)):
        if i == n: out = fr.to_ndarray(format="bgr24"); c.close(); return out
orb = cv2.ORB_create(3000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
def shift(a, b):
    # static-background mask: left strip (wall/power strip) + top-left; excludes arm / plate / wipes
    h, w = a.shape[:2]
    m = np.zeros((h, w), np.uint8); m[:, :int(0.28*w)] = 255
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ka, da = orb.detectAndCompute(ga, m); kb, db = orb.detectAndCompute(gb, m)
    if da is None or db is None: return None
    ms = bf.match(da, db)
    if len(ms) < 12: return None
    pa = np.float32([ka[x.queryIdx].pt for x in ms]); pb = np.float32([kb[x.trainIdx].pt for x in ms])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3)
    if M is None: return None
    return M[0,2], M[1,2], int(inl.sum()), np.degrees(np.arctan2(M[1,0], M[0,0]))
ref = train_f0(0)
print("Train ep k vs train ep0 (static left strip), dx,dy in px of the stored frame:")
for ep in [5, 20, 35, 50, 70]:
    print(f"  ep{ep:02d}:", shift(ref, train_f0(ep)))
print("Eval f0 vs train ep0:")
for run in ["eval_turbovla_smooth_success_1656", "eval_turbovla_warm_full_smooth", "eval_turbovla_warm_body_smooth", "eval_turbovla_smooth"]:
    vs = sorted(glob.glob(f"/media/yunyi/Data/Desktop/work/vla/eval_runs/local/{run}/videos/observation.images.top/**/*.mp4", recursive=True))
    print(f"  {run:40s}", shift(ref, dec(vs[0], 0)))
print("image size:", ref.shape)

import sys, glob, av, numpy as np, cv2, pandas as pd
sys.path.insert(0, "/media/yunyi/Data/Desktop/work/vla/TurboVLA/experiments/wipes")
from wipes_dataset import LeRobotWipesDataset
ds = LeRobotWipesDataset(dataset_dir="/media/yunyi/Data/Desktop/work/vla/datasets/wipes-to-green-plate-01",
    stats_path="/media/yunyi/Data/Desktop/work/vla/weights/turbovla_wipes/wipes_stats.json")
meta = ds.lerobot_ds.meta.episodes

def analyze(img_bgr):
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    g = cv2.inRange(hsv, (40, 120, 80), (90, 255, 255))
    ys, xs = np.nonzero(g)
    plate = (xs.mean()/w, ys.mean()/h, len(xs)) if len(xs) > 200 else (np.nan, np.nan, 0)
    # wipes: bright low-saturation blob on table: x>0.30 (exclude white wall/power strip on the left), y>0.30
    wmask = cv2.inRange(hsv, (0, 0, 190), (180, 70, 255))
    wmask[:, :int(0.30*w)] = 0; wmask[:int(0.30*h), :] = 0
    wmask = cv2.morphologyEx(wmask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(wmask)
    best = None
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if 80 < a < 2500 and (best is None or a > stats[best, cv2.CC_STAT_AREA]): best = i
    wipe = (cent[best][0]/w, cent[best][1]/h) if best is not None else (np.nan, np.nan)
    return plate, wipe

rows = []
for ep in range(len(meta)):
    frm = int(meta[ep]["dataset_from_index"])
    im = (ds.lerobot_ds[frm]["observation.images.top"].permute(1,2,0).numpy()*255).astype(np.uint8)
    p, wp = analyze(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    rows.append((ep, p[0], p[1], wp[0], wp[1]))
df = pd.DataFrame(rows, columns=["ep","plate_x","plate_y","wipe_x","wipe_y"])
print("TRAIN frame0 (normalized image coords, 0..1)  n=%d" % len(df))
print(df[["plate_x","plate_y","wipe_x","wipe_y"]].describe().loc[["mean","std","min","max"]].round(3))
df.to_csv("/tmp/train_f0_pos.csv", index=False)

def dec(v, n=0):
    c = av.open(v)
    for i, fr in enumerate(c.decode(video=0)):
        if i == n:
            out = fr.to_ndarray(format="bgr24"); c.close(); return out
print("\nEVAL runs frame0 (top cam):")
for run in ["eval_turbovla_smooth_success_1656", "eval_turbovla_warm_full_smooth", "eval_turbovla_warm_body_smooth", "eval_turbovla_smooth"]:
    vs = sorted(glob.glob(f"/media/yunyi/Data/Desktop/work/vla/eval_runs/local/{run}/videos/observation.images.top/**/*.mp4", recursive=True))
    if not vs: print(run, "no video"); continue
    p, wp = analyze(dec(vs[0], 0))
    print(f"  {run:40s} plate=({p[0]:.3f},{p[1]:.3f}) wipe=({wp[0]:.3f},{wp[1]:.3f})")

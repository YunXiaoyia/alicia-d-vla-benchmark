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
vs = sorted(glob.glob("/media/yunyi/Data/Desktop/work/vla/eval_runs/local/eval_turbovla_smooth/videos/observation.images.top/**/*.mp4", recursive=True))
a = train_f0(0); b = train_f0(40); c = dec(vs[0], 0)
for im, t in [(a,"TRAIN ep0 f0"),(b,"TRAIN ep40 f0"),(c,"EVAL 00:42 f0")]:
    cv2.putText(im, t, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
cv2.imwrite("/home/yunyi/.gemini/antigravity/brain/f0804794-81db-41e6-8649-89b9b32099d9/scratch/train_vs_eval_top.jpg", np.hstack([a,b,c]))

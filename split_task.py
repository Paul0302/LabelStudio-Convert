import json
from pathlib import Path

# 整包 Label Studio 專案 JSON
input_json = "pp.json"   # 如果檔名不一樣這裡改掉

with open(input_json, "r", encoding="utf-8") as f:
    tasks = json.load(f)

print(f"讀到 {len(tasks)} 個 task")

out_dir = Path("tasks_json")
out_dir.mkdir(exist_ok=True)

for task in tasks:
    task_id = task.get("id")
    file_upload = task.get("file_upload")  # 影片檔名，例如 e03b28c2-xxxx.MP4
    # 取一個比較好懂的檔名：taskID_原影片名.json
    safe_name = file_upload.replace("/", "_") if file_upload else f"task_{task_id}"
    out_path = out_dir / f"{task_id}_{safe_name}.json"

    # 這個 repo 期待的格式是「list」，所以外面包一層 []
    with open(out_path, "w", encoding="utf-8") as out_f:
        json.dump([task], out_f, ensure_ascii=False, indent=2)

    print(f"輸出: {out_path}")

print("全部切完！")

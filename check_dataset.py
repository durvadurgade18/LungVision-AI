import os

for split in ["train", "val", "test"]:
    print(f"\n===== {split.upper()} =====")

    base = os.path.join("DATASET", split)

    for cls in sorted(os.listdir(base)):
        path = os.path.join(base, cls)

        if os.path.isdir(path):
            count = len([
                f for f in os.listdir(path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            print(f"{cls:25} {count}")
# -*- coding: utf-8 -*-
"""Умный сортировщик оборудования — локальное Tkinter-приложение."""
from __future__ import annotations
import json, os, shutil, threading, traceback
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from core import classify_embedding, iter_image_files

APP_NAME = "Умный сортировщик оборудования"
MODEL_ID = "openai/clip-vit-base-patch32"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
BASE = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "SmartEquipmentSorter"
EXAMPLES = BASE / "examples"
CONFIG = BASE / "config.json"
DEFAULT_CATEGORIES = ["Другое", "На проверку"]

def load_config():
    BASE.mkdir(parents=True, exist_ok=True)
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    if CONFIG.exists():
        try:
            data = json.loads(CONFIG.read_text(encoding="utf-8"))
            data.setdefault("categories", DEFAULT_CATEGORIES[:])
            data.setdefault("examples", {})
            data.setdefault("threshold", 0.58)
            data.setdefault("history", [])
            return data
        except Exception:
            pass
    return {"categories": DEFAULT_CATEGORIES[:], "examples": {}, "threshold": 0.58, "history": []}

def save_config(data):
    BASE.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

class ClipEngine:
    def __init__(self):
        self.model = None
        self.processor = None
        self.torch = None

    def load(self, status):
        if self.model is not None:
            return
        status("Загрузка CLIP. При первом запуске требуется интернет…")
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        cache = str(BASE / "model_cache")
        Path(cache).mkdir(parents=True, exist_ok=True)
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID, cache_dir=cache)
        self.model = CLIPModel.from_pretrained(MODEL_ID, cache_dir=cache)
        self.model.eval()
        status("Модель загружена. Распознавание выполняется локально.")

    def embed(self, image_path):
        from PIL import Image, ImageOps
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        # CLIP's standard center crop can cut off a large attachment near an edge.
        # Compare both its standard view and a full-frame, aspect-ratio-preserving view.
        full_frame = ImageOps.pad(
            image, (224, 224), method=Image.Resampling.LANCZOS, color=(127, 127, 127)
        )
        inputs = self.processor(images=[image, full_frame], return_tensors="pt")
        with self.torch.inference_mode():
            vectors = self.model.get_image_features(**inputs)
            vectors = vectors / vectors.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            vector = vectors.mean(dim=0, keepdim=True)
            vector = vector / vector.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        return vector[0].cpu().tolist()

def cosine(a, b):
    return sum(x*y for x, y in zip(a, b))

class SorterApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("1120x720")
        self.root.minsize(900, 580)
        self.data = load_config()
        self.engine = ClipEngine()
        self.results = []
        self.source_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.status = tk.StringVar(value="Добавьте фото-образцы для категорий, затем выберите папку.")
        self.busy = False
        self._style()
        self._build()
        self.refresh_categories()

    def _style(self):
        style = ttk.Style()
        try: style.theme_use("vista")
        except tk.TclError: pass
        style.configure("Treeview", rowheight=28, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        style.configure("TButton", padding=6)

    def _build(self):
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")
        ttk.Label(top, text=APP_NAME, font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(top, text="Сортировка по визуальным образцам • локальная обработка • исходники не изменяются",
                  foreground="#555").pack(anchor="w", pady=(2, 8))
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.tab_samples = ttk.Frame(self.tabs, padding=10)
        self.tab_sort = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(self.tab_samples, text="  Образцы и категории  ")
        self.tabs.add(self.tab_sort, text="  Анализ и сортировка  ")
        self._build_samples()
        self._build_sort()
        bottom = ttk.Frame(self.root, padding=(12, 4, 12, 10))
        bottom.pack(fill="x")
        self.progress = ttk.Progressbar(bottom, mode="determinate")
        self.progress.pack(fill="x", pady=(0, 5))
        ttk.Label(bottom, textvariable=self.status).pack(anchor="w")

    def _build_samples(self):
        pane = ttk.Panedwindow(self.tab_samples, orient="horizontal")
        pane.pack(fill="both", expand=True)
        left = ttk.Frame(pane, padding=5); right = ttk.Frame(pane, padding=5)
        pane.add(left, weight=1); pane.add(right, weight=2)
        ttk.Label(left, text="Категории", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.cat_list = tk.Listbox(left, height=18, font=("Segoe UI", 10), exportselection=False)
        self.cat_list.pack(fill="both", expand=True, pady=6)
        self.cat_list.bind("<<ListboxSelect>>", lambda e: self.show_examples())
        row = ttk.Frame(left); row.pack(fill="x")
        ttk.Button(row, text="Новая категория", command=self.add_category).pack(side="left")
        ttk.Button(row, text="Удалить", command=self.delete_category).pack(side="left", padx=5)
        ttk.Label(right, text="Фотографии-образцы выбранной категории", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.examples_list = tk.Listbox(right, height=18, font=("Segoe UI", 9))
        self.examples_list.pack(fill="both", expand=True, pady=6)
        sample_actions = ttk.Frame(right); sample_actions.pack(anchor="w")
        ttk.Button(sample_actions, text="Добавить фото-образцы", command=self.add_examples).pack(side="left")
        ttk.Button(sample_actions, text="Удалить выбранный образец", command=self.delete_example).pack(side="left", padx=6)
        ttk.Label(right, text="Совет: добавляйте разные модели, цвета и ракурсы одного типа оборудования.",
                  foreground="#555", wraplength=600).pack(anchor="w", pady=10)

    def _build_sort(self):
        paths = ttk.LabelFrame(self.tab_sort, text="Папки", padding=8)
        paths.pack(fill="x", pady=(0, 8))
        self._path_row(paths, "Исходная папка:", self.source_dir, self.pick_source)
        self._path_row(paths, "Папка результата:", self.output_dir, self.pick_output)
        actions = ttk.Frame(self.tab_sort); actions.pack(fill="x", pady=(0, 8))
        self.threshold_var = tk.DoubleVar(value=float(self.data.get("threshold", .58)))
        ttk.Label(actions, text="Порог сходства:").pack(side="left")
        ttk.Scale(actions, from_=0.25, to=0.90, variable=self.threshold_var, length=180).pack(side="left", padx=6)
        self.threshold_label = ttk.Label(actions, text=f"{self.threshold_var.get():.2f}")
        self.threshold_label.pack(side="left")
        self.threshold_var.trace_add("write", lambda *_: self.threshold_label.config(text=f"{self.threshold_var.get():.2f}"))
        ttk.Button(actions, text="Начать анализ", command=self.start_analysis).pack(side="left", padx=12)
        ttk.Button(actions, text="Копировать подтверждённые", command=self.copy_results).pack(side="left")
        ttk.Button(actions, text="Сохранить изменения", command=self.save_manual).pack(side="left", padx=6)
        columns = ("file", "category", "score", "state")
        self.tree = ttk.Treeview(self.tab_sort, columns=columns, show="headings", selectmode="browse")
        for col, title, width in [("file","Файл",460),("category","Категория",190),("score","Сходство CLIP",110),("state","Статус",140)]:
            self.tree.heading(col, text=title); self.tree.column(col, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", self.change_selected)
        ttk.Label(self.tab_sort, text="Дважды щёлкните по строке, чтобы открыть фото и изменить категорию.",
                  foreground="#555").pack(anchor="w", pady=(6, 0))

    def _path_row(self, parent, label, var, command):
        row = ttk.Frame(parent); row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=20).pack(side="left")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(row, text="Обзор…", command=command).pack(side="left")

    def refresh_categories(self):
        self.cat_list.delete(0, "end")
        for c in self.data["categories"]: self.cat_list.insert("end", c)
        if self.data["categories"]:
            self.cat_list.selection_set(0); self.show_examples()

    def selected_category(self):
        sel = self.cat_list.curselection()
        return self.cat_list.get(sel[0]) if sel else None

    def show_examples(self):
        cat = self.selected_category()
        self.examples_list.delete(0, "end")
        for p in self.data["examples"].get(cat or "", []):
            self.examples_list.insert("end", Path(p).name)

    def add_category(self):
        name = simpledialog.askstring("Новая категория", "Название категории:", parent=self.root)
        if not name: return
        name = name.strip()
        if name and name not in self.data["categories"]:
            self.data["categories"].insert(max(0, len(self.data["categories"]) - 2), name)
            self.data["examples"].setdefault(name, [])
            save_config(self.data); self.refresh_categories()

    def delete_category(self):
        cat = self.selected_category()
        if not cat or cat in ("Другое", "На проверку"): return
        if messagebox.askyesno("Удаление", f"Удалить категорию «{cat}» и её список образцов?"):
            self.data["categories"].remove(cat)
            self.data["examples"].pop(cat, None)
            save_config(self.data); self.refresh_categories()

    def add_examples(self):
        cat = self.selected_category()
        if not cat: return
        paths = filedialog.askopenfilenames(title="Выберите фото-образцы",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.webp")])
        if not paths: return
        folder = EXAMPLES / str(abs(hash(cat)))
        folder.mkdir(parents=True, exist_ok=True)
        known = self.data["examples"].setdefault(cat, [])
        for src in paths:
            dst = folder / Path(src).name
            n = 1
            while dst.exists():
                dst = folder / f"{Path(src).stem}_{n}{Path(src).suffix}"; n += 1
            shutil.copy2(src, dst); known.append(str(dst))
        save_config(self.data); self.show_examples()
        self.status.set(f"Добавлено образцов: {len(paths)} для категории «{cat}».")

    def delete_example(self):
        cat = self.selected_category()
        selection = self.examples_list.curselection()
        if not cat or not selection:
            messagebox.showinfo("Образец не выбран", "Выберите фотографию образца в списке.")
            return
        paths = self.data["examples"].get(cat, [])
        index = selection[0]
        if index >= len(paths):
            return
        path = paths[index]
        if not messagebox.askyesno("Удаление образца", f"Удалить образец «{Path(path).name}» из категории «{cat}»?"):
            return
        paths.pop(index)
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass
        save_config(self.data)
        self.show_examples()
        self.status.set(f"Образец удалён из категории «{cat}».")

    def pick_source(self):
        p = filedialog.askdirectory(title="Выберите папку с фотографиями")
        if p: self.source_dir.set(p)

    def pick_output(self):
        p = filedialog.askdirectory(title="Выберите папку для результата")
        if p: self.output_dir.set(p)

    def set_status(self, text):
        self.root.after(0, self.status.set, text)

    def start_analysis(self):
        if self.busy: return
        source = Path(self.source_dir.get())
        if not source.is_dir():
            messagebox.showwarning("Папка не выбрана", "Выберите существующую исходную папку."); return
        target_text = self.output_dir.get().strip()
        if target_text:
            target = Path(target_text).expanduser().resolve()
            if target == source.resolve():
                messagebox.showwarning("Папки совпадают", "Папка результата не может совпадать с исходной папкой.")
                return
        samples = {c: list(self.data["examples"].get(c, [])) for c in self.data["categories"] if self.data["examples"].get(c)}
        if not samples:
            messagebox.showwarning("Нет образцов", "Сначала добавьте хотя бы один фото-образец."); self.tabs.select(self.tab_samples); return
        self.data["threshold"] = float(self.threshold_var.get()); save_config(self.data)
        self.busy = True
        threading.Thread(target=self._analyze_worker, args=(source, samples), daemon=True).start()

    def _analyze_worker(self, source, samples):
        try:
            self.engine.load(self.set_status)
            vectors = {}
            for cat, paths in samples.items():
                vectors[cat] = []
                for p in paths:
                    if Path(p).is_file():
                        vectors[cat].append(self.engine.embed(p))
            excluded = [BASE]
            if self.output_dir.get().strip():
                excluded.append(Path(self.output_dir.get()).expanduser().resolve())
            files = list(iter_image_files(source, IMAGE_EXTS, excluded))
            self.results = []
            self.root.after(0, self.progress.configure, {"maximum": max(1, len(files)), "value": 0})
            threshold = float(self.data.get("threshold", .58))
            for i, path in enumerate(files, 1):
                try:
                    v = self.engine.embed(path)
                    category, best_score, margin, ranked = classify_embedding(
                        v, vectors, threshold=threshold, min_margin=0.035
                    )
                    state = "Готово"
                    if category == "На проверку" and ranked:
                        state = "Низкое сходство или неоднозначный результат"
                    self.results.append({
                        "path": str(path), "category": category, "score": best_score,
                        "margin": margin, "state": state
                    })
                except Exception as e:
                    self.results.append({"path":str(path), "category":"На проверку", "score":0.0, "state":f"Ошибка: {str(e)[:45]}"})
                if i % 5 == 0 or i == len(files):
                    self.root.after(0, self.progress.configure, {"value":i})
                    self.set_status(f"Обработано {i} из {len(files)} фотографий…")
            self.root.after(0, self.show_results)
            self.set_status(f"Анализ завершён. Найдено фотографий: {len(files)}. Проверьте результаты перед копированием.")
        except Exception:
            self.set_status("Ошибка анализа: " + traceback.format_exc(limit=1))
        finally:
            self.busy = False

    def show_results(self):
        self.tree.delete(*self.tree.get_children())
        for i, r in enumerate(self.results):
            score = "—" if r["score"] < 0 else f"{r['score']:.3f}"
            self.tree.insert("", "end", iid=str(i), values=(Path(r["path"]).name, r["category"], score, r["state"]))
        self.tabs.select(self.tab_sort)

    def change_selected(self, _event=None):
        sel = self.tree.selection()
        if not sel: return
        idx = int(sel[0]); r = self.results[idx]
        try: os.startfile(r["path"])
        except Exception: pass
        choices = [c for c in self.data["categories"] if c != "На проверку"]
        choice = simpledialog.askstring("Изменить категорию",
            "Введите категорию из списка:\n" + ", ".join(choices), initialvalue=r["category"], parent=self.root)
        if choice and choice in self.data["categories"] and choice != "На проверку":
            r["category"] = choice; r["state"] = "Изменено вручную"
            self.tree.set(str(idx), "category", choice); self.tree.set(str(idx), "state", r["state"])

    def save_manual(self):
        for i, r in enumerate(self.results):
            vals = self.tree.item(str(i), "values") if self.tree.exists(str(i)) else None
            if vals: r["category"] = vals[1]
        self.status.set("Ручные изменения сохранены в текущем списке результатов.")

    def copy_results(self):
        if not self.results: messagebox.showinfo("Нет результатов", "Сначала выполните анализ."); return
        target = Path(self.output_dir.get())
        if not self.output_dir.get():
            messagebox.showwarning("Папка назначения", "Выберите папку результата."); return
        approved = [r for r in self.results if r["category"] != "На проверку" and not str(r["state"]).startswith("Ошибка")]
        if not approved:
            messagebox.showinfo("Нет подтверждённых результатов", "Все фотографии находятся на проверке либо содержат ошибки.")
            return
        if not messagebox.askyesno("Подтверждение", f"Скопировать {len(approved)} подтверждённых фотографий? Остальные останутся без копирования. Исходники не изменяются."):
            return
        copied = 0
        for r in approved:
            cat = r["category"]
            folder = target / cat; folder.mkdir(parents=True, exist_ok=True)
            src = Path(r["path"]); dst = folder / src.name
            n = 1
            while dst.exists():
                dst = folder / f"{src.stem}_{n}{src.suffix}"; n += 1
            try: shutil.copy2(src, dst); copied += 1
            except Exception: pass
        self.data["history"].append({"date":datetime.now().isoformat(timespec="seconds"), "count":copied, "source":self.source_dir.get(), "output":str(target)})
        save_config(self.data)
        messagebox.showinfo("Готово", f"Скопировано файлов: {copied}\nИсходные фотографии не изменялись.")
        self.status.set(f"Копирование завершено: {copied} файлов.")

def main():
    root = tk.Tk()
    SorterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

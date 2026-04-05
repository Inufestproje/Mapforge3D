# Sample Usage

This folder contains quick example commands so a new user can produce STL output immediately.

## 1) Point + Radius (Quick Test)

```bash
python main.py point-radius 38.35 38.32 600 --target-size-mm 120
```

## 2) Place + Radius

```bash
python main.py place-radius "Taksim Meydani, Istanbul" 1200 --target-size-mm 160
```

## 3) Polygon Input

Use the polygon string from `example_polygon.txt`:

```bash
python main.py polygon "41.037,28.986; 41.040,28.992; 41.033,28.997; 41.031,28.985"
```

Output STL files are written to:

`3_boyutlu_stl_dosyalari/`

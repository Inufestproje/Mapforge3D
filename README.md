# osm-to-stl

OpenStreetMap verilerini topografya ile birlestirip 3B baskiya uygun STL sahnesi ureten Python uygulamasi.

## Ne Uretiyor

- bina ayak izleri ve yukseklikleri
- yollar
- demiryollari
- nehir, kanal, gol gibi su katmanlari
- park ve yesil alanlar
- arazi engebesi, tepeler ve dag formu

Tum katmanlar tek bir birlesik STL dosyasina aktarilir.

## Desteklenen Alan Tipleri

- `place-radius`: bir yer adi etrafinda dairesel alan
- `place-boundary`: yer adinin idari siniri (tum sehir/il/provins vb.)
- `point-radius`: lat/lon etrafinda dairesel alan
- `bbox`: dikdortgen alan
- `polygon`: koordinat listesiyle serbest cokgen alan

## Kurulum

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Hizli kontrol:

```bash
python main.py --help
```

Not:

- `venv/` repoya yuklenmez, her kullanici kendi ortaminda olusturur.
- `cache/` repoya yuklenmez, calisma sirasinda otomatik olusur.
- STL ciktilari calisma sirasinda `3_boyutlu_stl_dosyalari/` altina yazilir.

## Ana Kullanim

### 1. Nokta + yaricap

```bash
python main.py point-radius 38.35 38.32 1500
```

### 2. Yer adi + yaricap

```bash
python main.py place-radius "Taksim Meydani, Istanbul" 1200
```

### 3. Dikdortgen alan

```bash
python main.py bbox 41.045 41.035 28.995 28.975
```

### 4. Yer adinin idari siniri

```bash
python main.py place-boundary "Malatya, Turkiye"
```

### 5. Koordinat cokgeni

```bash
python main.py polygon "41.037,28.986; 41.040,28.992; 41.033,28.997; 41.031,28.985"
```

Malatya topo + demiryolu odakli ornek:

```bash
python main.py place-boundary "Malatya, Turkiye" \
  --without-buildings \
  --without-roads \
  --without-water \
  --without-parks \
  --print-profile preview \
  --min-feature-width-mm 0.04 \
  --min-feature-area-mm2 0.002 \
  --terrain-zoom 9 \
  --target-size-mm 260
```

## Faydali Opsiyonlar

```bash
python main.py point-radius 38.35 38.32 800 \
  --max-buildings 200 \
  --target-size-mm 180 \
  --terrain-zoom 13 \
  --terrain-max-size 220
```

### Baski profilleri

```bash
python main.py place-radius "Times Square, New York" 400 --print-profile balanced
python main.py place-radius "Times Square, New York" 400 --print-profile fdm
python main.py place-radius "Times Square, New York" 400 --print-profile resin
python main.py place-radius "Times Square, New York" 400 --print-profile fdm --terrain-embed-mm 0.45
```

- `balanced`: genel kullanim icin dengeli gorunum
- `fdm`: daha kalin agac, yol, demiryolu, park ve cati detaylari ile FDM yaziciya uygun profil
- `resin`: daha ince ve detay odakli profil
- `--terrain-embed-mm`: bina, yol ve diger katmanlari tabana hafif gomerek bosluk goruntusunu azaltir
- `--railway-height-mm`: demiryolu katmaninin yuksekligini manuel ayarlar

Katman kapatma ornekleri:

```bash
python main.py point-radius 38.35 38.32 800 --without-water
python main.py point-radius 38.35 38.32 800 --without-parks
python main.py point-radius 38.35 38.32 800 --without-roads
python main.py point-radius 38.35 38.32 800 --without-railways
```

## Ciktilar

- varsayilan cikti klasoru: `3_boyutlu_stl_dosyalari/`
- birlesik sahne adi: `scene_<mod>_<alan>.stl`
- onbellek klasoru: `cache/`
- `cache/http/`: OSMnx HTTP cache
- `cache/features/`: katman bazli GeoJSON cache

## Yardimci Betikler

- `build_3d.py`: ornek bina + arazi ciktisi
- `terrain_buildings.py`: ornek arazi + bina ciktisi
- `merge_model.py`: ornek tam sahne ciktisi
- `fetch_buildings.py`: ornek yer adi tabanli cikti

Bu betiklerin hepsi arka planda yeni `main.py` akisina baglidir.

## Kullanilan Kutuphaneler

- `osmnx`
- `geopandas`
- `shapely`
- `trimesh`
- `numpy`
- `mercantile`
- `requests`
- `Pillow`
- `pyproj`
- `scipy` (varsa terrain yumusatma icin)

## Notlar

- Bina footprint detaylari korunur; girinti ve cikintilar dogrudan OSM geometrisinden gelir.
- Cok kucuk detaylar baskida kaybolmasin diye minimum alan ve minimum genislik esigi uygulanir.
- `fdm` profili, standart 0.4 mm nozzle gibi yazicilarda daha guvenli baski icin detaylari kalinlastirir.
- Su veya park verisi olmayan alanlarda bu katmanlar bos gecilir, islem hata vermez.

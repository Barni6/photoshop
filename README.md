# Photoshop Alkalmazás

Ez a projekt a **Képfeldolgozás alap algoritmusai** kurzus keretében készült, és egy grafikus képfeldolgozó alkalmazást valósít meg. Az alkalmazás különböző képmanipulációs feladatokat hajt végre, saját kézzel írt metódusok segítségével, a beépített könyvtári funkciók helyett.

## Funkciók

### Alapvető funkciók
- **Kép betöltése és visszaállítása**: Képek betöltése és eredeti állapotukra való visszaállítása.
- **Negálás**: Kép inverz színezése.
- **Gamma transzformáció**: Kép fényességének szabályozása gamma érték alapján.
- **Logaritmikus transzformáció**: Képkontraszt növelése logaritmikus függvény segítségével.
- **Szürkítés**: RGB kép szürkeárnyalatossá alakítása.

### Hisztogram funkciók
- **Hisztogram készítés**: Pixelintenzitás-eloszlás vizualizációja.
- **Hisztogram kiegyenlítés**: Kép dinamikatartományának kiegyenlítése saját algoritmussal.

### Szűrés és éldetektálás
- **Átlagoló szűrő**: Zaj csökkentése egyszerű átlagolással.
- **Gauss szűrő**: Zaj csökkentése Gauss eloszlás alapján.
- **Sobel éldetektor**: Kép éleinek detektálása Sobel operátor segítségével.
- **Laplace éldetektor**: Kép éleinek detektálása Laplace operátorral.

### Haladó funkciók
- **Jellemzőpontok detektálása**: ORB algoritmus használata kulcspontok felismerésére és vizualizálására.

## Technológiák

### Nyelv
- **Python**

### Könyvtárak
- **OpenCV**: Képfeldolgozáshoz.
- **Pillow**: Képek megjelenítéséhez.
- **Matplotlib**: Hisztogramok ábrázolásához.
- **Tkinter**: Grafikus felhasználói felülethez.



## Telepítési Útmutató

1. **Követelmények telepítése**:
   Telepítsd a szükséges Python könyvtárakat:
   ```bash
   pip install opencv-python-headless pillow matplotlib numpy

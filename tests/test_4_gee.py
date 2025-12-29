import ee
import os

def test_gee_connection():
    print("\n🧪 [4/4] TESTING GOOGLE EARTH ENGINE CONNECTION")
    print("="*40)
    
    try:
        # 1. Cek Inisialisasi
        ee.Initialize()
        print("   ✅ GEE Authentication Valid.")
        
        # 2. Test Request Ringan
        # Ambil 1 poin geometry sembarang di Lampung
        point = ee.Geometry.Point([105.0, -5.0])
        
        # Coba panggil koleksi Sentinel-2 (hanya metadata, 1 gambar)
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(point) \
            .filterDate('2021-01-01', '2021-01-30') \
            .first()
            
        info = s2.getInfo() # Ini request ke server Google
        print(f"   ✅ Berhasil mengambil metadata Sentinel-2 ID: {info['id']}")
        print("   ✅ Koneksi Internet & API Stabil.")
        
    except Exception as e:
        print(f"   ❌ GEE ERROR: {e}")
        print("      Coba jalankan 'earthengine authenticate' di terminal.")

if __name__ == "__main__":
    test_gee_connection()

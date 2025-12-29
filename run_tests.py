import os

if __name__ == "__main__":
    print("🛡️  MEMULAI SYSTEM WIDE CHECK SKRIPSI AI...")
    
    tests = [
        "tests/test_1_model_arch.py",
        "tests/test_2_dataset.py",
        "tests/test_3_overfit.py",
        "tests/test_4_gee.py"
    ]
    
    for t in tests:
        exit_code = os.system(f"python {t}")
        if exit_code != 0:
            print(f"\n❌ TERHENTI: Error ditemukan pada {t}")
            break
            
    print("\n🎉 SELESAI. Jika semua centang hijau, Anda siap training!")

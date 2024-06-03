# Project Retina Vessel Segmentation PCD

---

Program Segmentasi Retina untuk mencari Partition dari retina untuk dapat melakukan matching berdasarkan database

### Instalasi dan Setup

Untuk dapat menjalankan projek, pengguna perlu melakukan beberapa langkah penginstalan dan konfigurasi, agar aplikasi ini bisa berjalan dengan baik. berikut langkah-langkah yang perlu dilakukan.

1. **Klon repositori ke mesin lokal**

   Untuk melakukan klon repository, pengguna bisa menjalankan command berikut di terminal komputer sesuai sistem operasi yang digunakan.

   ```
   git clone https://github.com/depuntj/Retina_Vessel.git
   ```

2. **Masuk ke folder projek**

   Setelah repository berhasil dikloning, pengguna bisa masuk ke folder tersebut, dengan mengetikkan command berikut.

   ```
   cd .\Retina_Vessel\
   ```

3. **Install paket yang diperlukan**

   Sebelum paket-paket yang diperlukan belum terinstal, maka program tidak akan bisa dijalankan di komputer pengguna, sehingga pengguna perlu menginstal beberapa paket yang diperlukan. Untuk itu pengguna bisa mengetikkan command berikut.

   ```
   pip install -r requirements.txt
   ```

4. **Rubah image_path**
   Saat menjalan kan image processing kita perlu merubah gambar yang dinginkan untuk menjalankan image processing tersebut dan jalan kan main.ipynb
   ```
   python -u "ui.py"
   ```
5. **Jalnkan ui.py**
   Setelah melakukan itu semua jalankan ui.py untuk dapat melakukan matching dengan gambar yang dinginkan apakah ada didalam database yang di dapatkan dari menjalankan main.ipynb

   ```
   python -u "ui.py"
   ```

# =====================================================
# ADD 10 MORE AI TEXTS (GPT - BATCH 2)
# =====================================================

import pandas as pd
import sys

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Load existing data
OUTPUT_FILE = 'data_ai_gpt_10.csv'
try:
    df_existing = pd.read_csv(OUTPUT_FILE)
    existing_data = df_existing.to_dict('records')
    print(f"[INFO] Existing data: {len(existing_data)}")
except:
    existing_data = []
    print(f"[INFO] No existing data, creating new file")

# New texts from user
raw_texts = """---TEKS 1---
TOPIK: Sosial

Aku ngerasa kehidupan sosial sekarang tuh unik banget. Kita bisa ngobrol sama orang jauh lewat layar, tapi sama orang dekat malah jarang nyapa. Pernah gak sih kamu duduk bareng temen, tapi suasananya sunyi karena semua fokus ke HP masing-masing? Aku sering ngalamin itu, dan rasanya agak aneh. Kita bareng, tapi gak benar-benar bersama.

Menurutku, masalah sosial bukan karena kita jadi cuek, tapi karena terlalu terbiasa multitasking. Kita pengin denger cerita orang, tapi sambil scroll. Pengin respon, tapi setengah hati. Akhirnya obrolan jadi hambar dan gak nyambung. Padahal hubungan sosial itu butuh kehadiran, bukan cuma fisik, tapi juga perhatian.

Aku mulai nyoba hal kecil, kayak naro HP pas lagi ngobrol serius. Hasilnya beda banget. Cerita jadi ngalir, tawa lebih lepas, dan rasanya lebih dekat. Hal-hal sederhana kayak gitu sebenernya bisa bikin hubungan lebih sehat.

Kita hidup di zaman cepat, tapi hubungan gak bisa dibangun dengan buru-buru. Kadang kita cuma perlu berhenti sebentar, dengerin, dan bener-bener ada buat satu sama lain.

---TEKS 2---
TOPIK: Budaya

Kalau ngomongin budaya, aku sering mikir kenapa banyak orang nganggep budaya itu kuno. Padahal kalau dipikir-pikir, budaya itu cerita tentang siapa kita dan dari mana kita berasal. Tanpa budaya, kita cuma kayak ikut arus tanpa identitas.

Aku dulu juga gak terlalu peduli sama budaya daerah. Baru pas dewasa, aku sadar banyak hal menarik yang selama ini aku anggap biasa. Mulai dari bahasa daerah, kebiasaan keluarga, sampai cara orang tua kita berinteraksi, itu semua bagian dari budaya.

Sekarang budaya gak harus ditampilkan dengan cara kaku. Banyak anak muda yang ngemas budaya jadi lebih modern, entah lewat musik, fashion, atau konten digital. Dan menurutku itu keren. Budaya jadi hidup, bukan cuma pajangan.

Menjaga budaya gak berarti nolak perubahan. Justru budaya bisa berkembang tanpa kehilangan makna. Selama kita masih menghargai dan gak malu sama budaya sendiri, identitas itu bakal tetap ada.

---TEKS 3---
TOPIK: Politik

Jujur aja, politik sering bikin pusing. Beritanya ribut, pendapat orang saling bertabrakan, dan suasananya panas. Aku dulu milih cuek karena ngerasa politik cuma bikin emosi. Tapi lama-lama aku sadar, mau cuek atau enggak, dampaknya tetep ke hidup kita.

Politik itu gak cuma soal pejabat atau pemilu. Keputusan politik ngaruh ke harga kebutuhan, pendidikan, transportasi, dan banyak hal lain yang kita rasain tiap hari. Jadi sebenernya politik itu dekat, cuma sering dibikin kelihatan jauh.

Yang bikin capek itu pas orang beda pendapat tapi gak mau saling denger. Padahal beda pilihan itu wajar. Kita bisa kok diskusi tanpa harus marah-marah. Politik harusnya jadi ruang tukar pikiran, bukan ajang saling serang.

Menurutku, cukup jadi warga yang peduli dan kritis. Gak harus vokal berlebihan, tapi juga gak tutup mata. Dengan sikap santai tapi sadar, kita bisa lebih bijak ngadepin dunia politik.

---TEKS 4---
TOPIK: Ekonomi

Ekonomi sering bikin kita mikir keras, apalagi soal uang. Aku ngerasa ekonomi itu bukan cuma angka, tapi soal kebiasaan. Mau penghasilan besar kalau pengeluarannya gak keatur, ya tetap aja bocor.

Aku pernah ngerasa gaji numpang lewat. Baru awal bulan, eh tahu-tahu udah habis. Setelah dicek, ternyata banyak pengeluaran kecil yang gak kerasa. Jajan, langganan ini itu, dan belanja impulsif.

Sekarang aku mulai belajar ngatur, walau belum sempurna. Setidaknya aku lebih sadar sebelum beli sesuatu. Aku tanya ke diri sendiri, ini butuh atau cuma pengin. Kadang jawabannya bikin mikir ulang.

Ekonomi yang sehat itu bukan soal kaya mendadak, tapi soal rasa aman. Bisa bayar kebutuhan, gak stres mikirin besok, dan punya pegangan buat masa depan. Semua bisa dimulai dari kebiasaan kecil yang konsisten.

---TEKS 5---
TOPIK: Teknologi

Teknologi sekarang udah kayak tangan kedua. Mau cari info, kerja, hiburan, semua lewat teknologi. Aku ngerasa hidup jadi lebih praktis, tapi juga lebih ramai. Notifikasi datang terus, bikin kepala penuh.

Aku suka teknologi karena ngebantu banyak hal. Tapi aku juga sadar, kalau gak diatur, teknologi bisa bikin capek sendiri. Pernah niat istirahat, tapi malah kejebak scrolling lama banget.

Menurutku, teknologi itu alat, bukan pengendali. Kita yang harus pegang kendali. Sesekali off dari layar itu perlu, biar pikiran bisa napas. Aku mulai nentuin waktu tanpa HP, walau awalnya berat.

Kalau dipakai dengan bijak, teknologi bisa bikin kita berkembang. Belajar hal baru, kerja lebih efisien, dan terhubung dengan banyak orang. Kuncinya ada di cara kita nggunakannya.

---TEKS 6---
TOPIK: Kesehatan

Kesehatan sering baru terasa penting pas kita udah ngerasa gak enak badan. Aku dulu sering nunda tidur, makan sembarangan, mikirnya nanti aja dibenerin. Tapi badan gak bisa dibohongin.

Sekarang aku lebih paham kalau kesehatan itu investasi. Bukan cuma soal olahraga, tapi juga soal pola hidup. Tidur cukup, minum air, dan ngatur stres itu sama pentingnya.

Aku gak langsung berubah drastis. Mulai dari hal kecil dulu, kayak tidur lebih awal dan gerak dikit tiap hari. Hasilnya kerasa, badan lebih enteng dan pikiran lebih tenang.

Sehat itu bukan tentang jadi sempurna, tapi tentang peduli sama diri sendiri. Kita sering perhatian ke orang lain, tapi lupa ke tubuh sendiri. Padahal tubuh itu rumah kita seumur hidup.

---TEKS 7---
TOPIK: Olahraga

Dulu aku nganggep olahraga itu kewajiban yang berat. Harus niat, harus capek. Tapi setelah dijalanin pelan-pelan, aku sadar olahraga bisa jadi aktivitas yang menyenangkan.

Olahraga bikin badan gerak dan pikiran lebih fresh. Setelah keringetan, rasanya lega. Walau capek, tapi capeknya beda. Aku jadi lebih gampang tidur dan mood juga lebih stabil.

Aku gak maksa diri buat olahraga berat. Cukup yang ringan tapi rutin. Jalan kaki, stretching, atau sekadar gerak di rumah. Yang penting konsisten.

Olahraga itu soal kebiasaan, bukan paksaan. Kalau udah nemu ritmenya, olahraga jadi bagian dari hidup, bukan beban.

---TEKS 8---
TOPIK: Pendidikan

Pendidikan sering disempitin jadi sekolah dan nilai. Padahal pendidikan jauh lebih luas dari itu. Aku ngerasa banyak pelajaran hidup yang gak pernah diajarin di kelas, tapi penting banget.

Belajar sekarang gak harus formal. Internet bikin semua orang bisa belajar apa aja. Tinggal niat dan konsisten. Aku sendiri sering belajar hal baru dari video dan artikel gratis.

Pendidikan juga soal cara berpikir. Bukan cuma hafal, tapi paham. Bisa nanya, bisa ragu, dan mau cari jawaban. Itu yang bikin kita berkembang.

Selama kita masih mau belajar, pendidikan gak pernah berhenti. Umur bukan penghalang, rasa ingin tahu yang bikin jalan terus.

---TEKS 9---
TOPIK: Lingkungan

Lingkungan sering dibahas, tapi jarang dipikirin serius. Padahal dampaknya ada di sekitar kita. Cuaca makin ekstrem, udara panas, dan sampah di mana-mana.

Aku mulai sadar kalau kebiasaan kecil punya pengaruh. Buang sampah sembarangan, pakai plastik sekali pakai, itu kelihatannya sepele tapi efeknya panjang.

Aku gak langsung jadi super peduli, tapi mulai dari hal kecil. Bawa botol minum sendiri, kurangi plastik, dan lebih sadar sama sekitar.

Jaga lingkungan itu bukan buat orang lain, tapi buat kita sendiri. Kalau alam rusak, kita juga yang kena. Jadi peduli lingkungan itu bentuk tanggung jawab ke masa depan.

---TEKS 10---
TOPIK: Entertainment

Entertainment itu bagian penting dari hidup. Aku ngerasa hiburan itu kayak tombol pause di tengah rutinitas. Nonton, denger musik, atau main game bisa bikin pikiran lebih santai.

Sekarang hiburan gampang banget diakses. Tinggal buka HP, semua ada. Tapi justru karena itu, kita harus bisa ngatur waktu. Hiburan yang kebanyakan malah bikin capek.

Aku pernah kebablasan nikmatin hiburan sampai lupa waktu. Akhirnya yang ada malah kurang tidur. Dari situ aku belajar, hiburan itu buat dinikmati secukupnya.

Kalau seimbang, entertainment bisa jadi sumber energi. Bikin kita balik ke aktivitas dengan mood lebih baik dan pikiran lebih ringan."""

# Parse new texts
texts = raw_texts.split('---TEKS ')[1:]  # Skip empty first element

for text_block in texts:
    lines = text_block.strip().split('\n')
    header = lines[0]  # e.g., "1---"

    # Find topic
    topic = "Unknown"
    for i, line in enumerate(lines):
        if line.startswith('TOPIK:'):
            topic = line.replace('TOPIK:', '').strip()
            # Text starts after the topic line
            content_start = i + 2 if i + 1 < len(lines) and lines[i+1] == '' else i + 1
            text_content = '\n'.join(lines[content_start:]).strip()
            break
    else:
        # No TOPIK found, take everything after header
        text_content = '\n'.join(lines[1:]).strip()

    word_count = len(text_content.split())
    char_count = len(text_content)

    existing_data.append({
        'text': text_content,
        'label': 'AI',
        'source': 'GPT_Casual',
        'topic': topic,
        'word_count': word_count,
        'char_count': char_count
    })

# Save all data
df = pd.DataFrame(existing_data)
df.to_csv(OUTPUT_FILE, index=False)

print("="*60)
print("ADD 10 MORE AI TEXTS (GPT - BATCH 2)")
print("="*60)
print(f"\n[OK] Saved: {OUTPUT_FILE}")
print(f"Total: {len(df)} data\n")

for i, row in df.iterrows():
    print(f"{i+1}. {row['topic']:12} - {row['word_count']:3} kata")

print("\n" + "="*60)

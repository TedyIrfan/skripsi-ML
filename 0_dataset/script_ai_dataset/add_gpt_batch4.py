# =====================================================
# ADD 10 GPT CASUAL TEXTS (BATCH 4)
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

    # Fix source names - change Gemini_Casual to GPT_Casual
    for item in existing_data:
        if item['source'] == 'Gemini_Casual':
            item['source'] = 'GPT_Casual'

    print(f"[INFO] Existing data: {len(existing_data)}")
    print(f"[INFO] Fixed source names")
except:
    existing_data = []
    print(f"[INFO] No existing data, creating new file")

# New GPT texts (10)
gpt_texts = """---TEKS 1---
TOPIK: Sosial
Aku makin sadar kalau hubungan sosial itu gak selalu tentang sering ketemu atau sering chat. Kadang ada temen yang jarang ngobrol, tapi sekali ngobrol langsung nyambung dan dalem. Ada juga yang tiap hari komunikasi, tapi rasanya kosong. Dari situ aku belajar kalau kedekatan itu soal kualitas, bukan intensitas.

Di dunia sekarang, kita gampang banget terhubung, tapi juga gampang kehilangan makna. Banyak obrolan cuma basa-basi, jarang yang bener-bener jujur. Aku sendiri kadang capek harus selalu terlihat baik-baik aja di depan orang lain.

Pelan-pelan aku belajar buat lebih selektif. Bukan berarti menjauh dari semua orang, tapi lebih ke jaga energi. Pilih orang yang bikin kita nyaman jadi diri sendiri, bukan yang bikin kita terus pura-pura.

Hubungan sosial yang sehat itu harusnya bikin kita berkembang, bukan terkuras. Kalau setelah ketemu seseorang kamu ngerasa lebih tenang, berarti hubungan itu layak dijaga.

---TEKS 2---
TOPIK: Budaya
Budaya itu gak selalu soal upacara besar atau pakaian adat. Menurutku budaya justru paling kerasa di kebiasaan kecil sehari-hari. Cara kita nyapa orang tua, cara kita duduk saat ngobrol, sampai cara kita makan bareng keluarga.

Aku pernah ngerasa kebiasaan di rumah itu biasa aja, tapi setelah lihat lingkungan lain, baru kerasa bedanya. Dari situ aku sadar, kebiasaan itu bukan cuma rutinitas, tapi warisan.

Sekarang tantangannya gimana caranya budaya tetap hidup di tengah perubahan zaman. Anak muda maunya simpel dan praktis, tapi budaya sering dianggap ribet. Padahal sebenernya bisa disesuaikan tanpa kehilangan nilai.

Menurutku, selama nilai utamanya masih dijaga, bentuknya bisa fleksibel. Budaya itu hidup, bukan benda mati. Dia bisa berubah, tapi akarnya tetap sama.

---TEKS 3---
TOPIK: Politik
Politik sering terasa melelahkan karena isinya konflik terus. Aku kadang mikir, kenapa sih politik jarang dibahas dengan tenang? Padahal dampaknya ke hidup kita nyata banget.

Aku mulai ngikutin politik bukan karena suka ributnya, tapi karena pengin ngerti. Semakin paham, semakin sadar kalau banyak hal yang sebelumnya aku anggap biasa ternyata ada kebijakan di baliknya.

Masalahnya, banyak orang langsung defensif pas ngomongin politik. Sedikit beda sudut pandang langsung dianggap musuh. Padahal diskusi itu harusnya bikin wawasan nambah, bukan emosi naik.

Menurutku, politik bakal lebih sehat kalau dibahas pakai logika dan empati. Kita boleh beda pilihan, tapi tetap satu tujuan, hidup yang lebih baik.

---TEKS 4---
TOPIK: Ekonomi
Ekonomi itu sering bikin orang ngerasa tertekan, apalagi kalau pengeluaran lebih gede dari pemasukan. Aku pernah ngerasa kerja capek-capek tapi uangnya cepat habis. Rasanya frustasi.

Dari situ aku mulai sadar pentingnya perencanaan. Bukan soal ribet ngitung, tapi soal sadar ke mana uang pergi. Kadang masalahnya bukan kurang penghasilan, tapi kebocoran kecil yang gak disadari.

Aku mulai nyoba lebih disiplin, walau pelan-pelan. Gak selalu berhasil, tapi setidaknya lebih terkontrol. Yang penting ada usaha buat berubah.

Ekonomi yang stabil itu bukan soal pamer, tapi soal ketenangan. Bisa hidup tanpa panik mikirin uang tiap waktu itu rasanya lega banget.

---TEKS 5---
TOPIK: Teknologi
Teknologi bikin hidup kita serba terhubung, tapi kadang bikin kita jauh dari sekitar. Aku pernah duduk di satu ruangan penuh orang, tapi semua fokus ke layar masing-masing. Sunyi tapi rame.

Aku gak nyalahin teknologinya, karena emang ngebantu banyak hal. Tapi aku sadar, kita perlu batasan. Kalau gak, waktu habis tanpa terasa.

Sekarang aku coba lebih sadar kapan harus online dan kapan harus berhenti. Gak selalu berhasil, tapi setidaknya ada niat. Rasanya lebih enak pas bisa fokus ke satu hal.

Teknologi itu alat, bukan tujuan. Kalau kita bisa ngatur, teknologi bakal bantu hidup, bukan ngambil alih hidup.

---TEKS 6---
TOPIK: Kesehatan
Kesehatan sering jadi prioritas terakhir, padahal harusnya yang pertama. Aku sering nunda istirahat demi kerja atau hiburan. Awalnya kuat, lama-lama badan protes.

Aku mulai sadar kalau tubuh itu punya batas. Kalau dipaksa terus, pasti ada konsekuensinya. Sehat itu bukan soal kuat-kuatan, tapi soal pinter jaga diri.

Pelan-pelan aku belajar dengerin sinyal tubuh. Capek ya istirahat, stres ya cari jeda. Gak semua harus diselesain sekarang juga.

Kesehatan itu soal keseimbangan. Selama kita mau peduli dan gak keras ke diri sendiri, badan dan pikiran bakal lebih bersahabat.

---TEKS 7---
TOPIK: Olahraga
Olahraga sering kalah sama rasa malas. Aku juga sering ngalamin. Niat ada, tapi eksekusi nol. Tapi setelah dipaksa dikit, ternyata olahraga gak seseram itu.

Olahraga bikin badan gerak dan pikiran lebih segar. Walau capek, tapi capeknya bikin puas. Rasanya beda sama capek karena stres.

Aku gak ngejar target muluk. Yang penting rutin. Sedikit tapi konsisten lebih baik daripada semangat sesaat.

Kalau udah terbiasa, olahraga jadi kebutuhan, bukan paksaan. Badan terasa lebih siap ngadepin aktivitas harian.

---TEKS 8---
TOPIK: Pendidikan
Pendidikan bukan cuma soal dapet nilai bagus. Menurutku pendidikan itu soal gimana kita menyikapi hidup. Bisa mikir jernih, bisa ngambil keputusan, dan bisa belajar dari kesalahan.

Aku ngerasa banyak pelajaran penting justru aku dapet dari pengalaman gagal. Dari situ aku belajar lebih banyak daripada dari teori.

Sekarang belajar bisa dari mana aja. Buku, internet, pengalaman orang lain. Tinggal kitanya mau atau enggak.

Selama kita mau terbuka buat belajar, pendidikan bakal terus jalan, bahkan tanpa ruang kelas.

---TEKS 9---
TOPIK: Lingkungan
Lingkungan sering jadi topik besar, tapi tindakannya kecil. Aku dulu mikir satu orang gak bakal ngaruh. Tapi sekarang aku sadar, perubahan selalu dimulai dari satu langkah kecil.

Hal sederhana kayak buang sampah di tempatnya atau ngurangin plastik itu udah bentuk kepedulian. Gak harus langsung ekstrem.

Kalau banyak orang mulai peduli, dampaknya bakal kerasa. Lingkungan yang bersih bikin hidup lebih nyaman.

Peduli lingkungan itu bukan tren, tapi tanggung jawab. Kita jaga alam, alam jaga kita.

---TEKS 10---
TOPIK: Entertainment
Entertainment itu cara kita ngasih jeda buat diri sendiri. Aku sering pakai hiburan buat ngilangin stres. Nonton, denger musik, atau main game bikin pikiran lebih santai.

Tapi aku juga belajar kalau hiburan kebanyakan bisa bikin lupa waktu. Niatnya rehat, malah capek sendiri.

Sekarang aku coba lebih seimbang. Hiburan secukupnya, tanggung jawab tetap jalan. Rasanya lebih enak.

Entertainment itu harusnya bantu kita balik ke rutinitas dengan energi baru, bukan bikin kita tenggelam."""

def parse_texts(raw_texts, source_name):
    texts = raw_texts.split('---TEKS ')[1:]
    results = []

    for text_block in texts:
        lines = text_block.strip().split('\n')

        # Find topic
        topic = "Unknown"
        for i, line in enumerate(lines):
            if line.startswith('TOPIK:'):
                topic = line.replace('TOPIK:', '').strip()
                content_start = i + 2 if i + 1 < len(lines) and lines[i+1] == '' else i + 1
                text_content = '\n'.join(lines[content_start:]).strip()
                break
        else:
            text_content = '\n'.join(lines[1:]).strip()

        word_count = len(text_content.split())
        char_count = len(text_content)

        results.append({
            'text': text_content,
            'label': 'AI',
            'source': source_name,
            'topic': topic,
            'word_count': word_count,
            'char_count': char_count
        })

    return results

# Parse new texts
gpt_data = parse_texts(gpt_texts, 'GPT_Casual')

# Add to existing data
existing_data.extend(gpt_data)

# Save all data
df = pd.DataFrame(existing_data)
df.to_csv(OUTPUT_FILE, index=False)

print("="*60)
print("ADD 10 GPT CASUAL TEXTS (BATCH 4)")
print("="*60)
print(f"\n[OK] Saved: {OUTPUT_FILE}")
print(f"Total: {len(df)} data\n")

# Count by source
source_counts = {}
for item in existing_data:
    src = item['source']
    source_counts[src] = source_counts.get(src, 0) + 1

print("Breakdown by source:")
for source, count in sorted(source_counts.items()):
    print(f"  {source}: {count}")

print(f"\nTotal: {len(df)} data")
print("\n" + "="*60)

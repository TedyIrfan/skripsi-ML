# =====================================================
# PARSE & SAVE 10 AI TEXTS (GPT - CASUAL STYLE)
# =====================================================

import pandas as pd
import re
import sys
import os

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Hapus file lama
if os.path.exists('data_ai_gpt_10.csv'):
    os.remove('data_ai_gpt_10.csv')
    print("[INFO] Deleted old file: data_ai_gpt_10.csv\n")

# Data dari user (CASUAL STYLE)
raw_texts = """---TEKS 1---
TOPIK: Sosial

Akhir-akhir ini aku sering kepikiran soal gimana cara kita berinteraksi satu sama lain. Rasanya dunia makin rame, tapi kok kadang terasa sepi ya. Kita punya banyak grup chat, banyak followers, tapi pas lagi butuh cerita beneran, malah bingung mau ke siapa. Aku pernah ngalamin sendiri, nongkrong bareng temen-temen tapi semua sibuk sama HP masing-masing. Ada di satu tempat, tapi gak benar-benar "hadir".

Menurutku, masalah sosial sekarang bukan soal kurang temen, tapi kurang koneksi yang nyata. Kita terbiasa ngobrol lewat layar, jadi pas harus ngobrol langsung malah kikuk. Padahal hal simpel kayak nanya kabar dengan tulus atau dengerin cerita orang lain tanpa nyela itu penting banget. Kadang orang cuma butuh didengerin, gak perlu solusi.

Aku juga ngerasa empati itu mulai luntur, apalagi di media sosial. Sedikit salah, langsung dihakimi rame-rame. Kita lupa kalau di balik layar itu ada manusia juga, punya perasaan, punya masalah sendiri. Harusnya kita bisa lebih santai dan bijak.

Mungkin solusinya gak muluk-muluk. Mulai dari hal kecil aja, kayak nyapa tetangga, ngobrol sama temen tanpa main HP, atau sekadar nanya "kamu lagi kenapa?". Hal-hal kecil gitu kalau dilakuin bareng-bareng, bisa bikin kehidupan sosial kita lebih hangat lagi.

---TEKS 2---
TOPIK: Budaya

Ngomongin budaya itu sebenernya seru, apalagi budaya kita sendiri. Indonesia tuh kaya banget, tapi jujur aja, kadang kita malah lebih hafal budaya luar dibanding budaya daerah sendiri. Aku aja baru sadar beberapa tahun terakhir kalau ternyata daerahku punya tarian dan tradisi unik yang dulu sering dianggap kuno.

Budaya sering dicap jadul, ribet, atau gak relevan sama anak muda. Padahal kalau dikemas dengan cara yang lebih santai, budaya bisa keliatan keren. Lihat aja sekarang, banyak anak muda yang mulai pakai batik dengan style modern atau ngangkat musik tradisional ke versi remix. Itu bukti kalau budaya bisa berkembang tanpa kehilangan jati diri.

Menurutku, masalahnya bukan di budayanya, tapi di cara ngenalinnya. Kalau dari kecil kita cuma disuruh hafalan tanpa dikasih konteks, ya wajar aja jadi gak tertarik. Coba kalau ceritanya dibikin lebih hidup, pasti beda rasanya.

Kita gak harus jadi ahli budaya buat ikut melestarikan. Cukup dengan menghargai, gak meremehkan, dan mau belajar sedikit demi sedikit. Bangga sama budaya sendiri itu gak bikin kita ketinggalan zaman, justru nunjukin kalau kita tau siapa diri kita sebenarnya.

---TEKS 3---
TOPIK: Politik

Jujur aja, dulu aku agak males denger kata politik. Kesannya ribet, penuh drama, dan kayak gak ada hubungannya sama hidup sehari-hari. Tapi makin ke sini, aku sadar kalau politik itu sebenernya dekat banget sama kita, cuma sering dibikin kelihatan jauh.

Hal-hal sederhana kayak harga makanan naik, aturan pendidikan, sampai fasilitas umum, itu semua hasil keputusan politik. Mau gak mau, kita kena dampaknya. Masalahnya, banyak dari kita yang apatis karena ngerasa suara kita gak didengerin. Padahal kalau semua mikir gitu, ya gak akan ada perubahan.

Politik juga sering panas karena orang terlalu fanatik. Sedikit beda pendapat langsung berantem, padahal beda pandangan itu wajar. Kita bisa kok gak setuju tanpa harus saling menjatuhkan. Diskusi santai malah lebih sehat daripada debat penuh emosi.

Menurutku, yang penting sekarang itu melek politik, bukan berarti harus ikut partai, tapi minimal paham dan kritis. Jangan gampang kemakan hoaks, cek info, dan berani nanya. Politik gak harus serem, kalau kita hadapin dengan kepala dingin dan niat buat kebaikan bareng-bareng.

---TEKS 4---
TOPIK: Ekonomi

Ngomongin ekonomi, hal pertama yang kepikiran biasanya uang. Dan jujur, siapa sih yang gak kepikiran soal itu? Dari bangun tidur sampai mau tidur lagi, ekonomi selalu ikut campur. Mau jajan mikir harga, mau liburan mikir budget, mau nabung tapi pengeluaran jalan terus.

Aku ngerasa tantangan ekonomi sekarang bukan cuma soal cari uang, tapi gimana ngaturnya. Banyak dari kita yang penghasilannya ada, tapi tetep ngerasa kurang. Kadang bukan karena kurang, tapi karena gak sadar ke mana aja uangnya pergi. Aku juga pernah ngalamin, tiba-tiba akhir bulan saldo tinggal sisa kenangan.

Sekarang banyak banget godaan buat belanja impulsif, apalagi dengan adanya promo dan paylater. Awalnya keliatan ringan, lama-lama numpuk. Makanya penting banget buat punya kesadaran finansial, walaupun pelan-pelan.

Ekonomi itu gak melulu soal angka besar. Mulai dari nyatet pengeluaran, bedain kebutuhan sama keinginan, sampai nyisihin sedikit buat tabungan. Gak harus langsung jago, yang penting konsisten. Sedikit demi sedikit, kondisi ekonomi kita bisa lebih sehat dan gak bikin stres tiap akhir bulan.

---TEKS 5---
TOPIK: Teknologi

Teknologi sekarang udah jadi bagian hidup kita, mau gak mau. Bangun tidur cek HP, mau tidur cek HP lagi. Aku kadang mikir, hidup kita sekarang tuh setengahnya ada di dunia digital. Teknologi emang ngebantu banget, bikin semua jadi cepat dan praktis.

Tapi di sisi lain, teknologi juga bikin kita gampang terdistraksi. Niatnya cuma buka HP lima menit, tau-tau sejam lewat. Aku sering ngerasa capek bukan karena kerja fisik, tapi karena kebanyakan nerima informasi. Otak kayak gak dikasih jeda.

Teknologi sebenernya netral, tergantung kita yang pake. Bisa jadi alat produktif, bisa juga jadi sumber masalah. Media sosial misalnya, bisa buat cari inspirasi, tapi juga bisa bikin kita ngebandingin diri sama orang lain terus.

Menurutku, yang penting itu bijak. Tau kapan harus online, kapan harus berhenti. Pakai teknologi buat bantu berkembang, bukan malah bikin kita stuck. Kalau kita bisa ngatur, teknologi bakal jadi temen yang ngebantu, bukan yang nguras energi.

---TEKS 6---
TOPIK: Kesehatan

Kesehatan sering dianggap sepele sampai akhirnya kita sakit. Aku sendiri dulu sering begadang, makan asal-asalan, mikirnya "ah masih muda". Tapi pas badan mulai gampang capek, baru kerasa kalau kesehatan itu mahal banget.

Gaya hidup sekarang emang bikin kita susah jaga kesehatan. Duduk kelamaan, kurang gerak, makanan cepat saji gampang banget ditemuin. Belum lagi stres yang datang dari berbagai arah. Semua itu pelan-pelan ngaruh ke tubuh.

Kesehatan gak selalu soal olahraga berat atau diet ketat. Hal kecil kayak cukup minum air, tidur teratur, dan jalan kaki sebentar aja udah ngaruh. Aku ngerasa badan lebih enak cuma dengan ngurangin begadang dan makan lebih teratur.

Yang paling penting, dengerin tubuh sendiri. Kalau capek, istirahat. Kalau stres, cari jeda. Kita sering maksa diri terus, padahal badan juga butuh perhatian. Sehat itu bukan tujuan instan, tapi proses yang harus dijalanin pelan-pelan.

---TEKS 7---
TOPIK: Olahraga

Olahraga itu sering kebayangnya capek dan ribet, padahal sebenernya bisa santai. Aku dulu termasuk yang males olahraga, alesannya klasik: gak ada waktu. Padahal sebenernya bukan gak ada waktu, tapi gak dibiasain.

Sekarang aku mulai sadar kalau olahraga itu bukan buat kelihatan keren aja, tapi buat bikin badan dan pikiran lebih enak. Setelah gerak, walau cuma sebentar, mood biasanya jadi lebih baik. Keringetan dikit, tapi rasanya lega.

Olahraga gak harus ke gym atau pake alat mahal. Jalan kaki, stretching, atau main bola sama temen juga termasuk olahraga. Yang penting badan bergerak dan dilakukan rutin.

Menurutku, kunci olahraga itu konsistensi, bukan intensitas. Lebih baik olahraga ringan tapi rutin, daripada berat tapi cuma sekali. Kalau udah nemu jenis olahraga yang kita suka, olahraga gak lagi jadi beban, tapi jadi kebiasaan yang dinantiin.

---TEKS 8---
TOPIK: Pendidikan

Pendidikan gak cuma soal sekolah dan nilai. Aku ngerasa pendidikan itu proses panjang yang gak berhenti pas kita lulus. Banyak hal penting dalam hidup yang justru gak diajarin di kelas, tapi kita pelajari dari pengalaman.

Sekolah emang penting buat dasar, tapi cara belajar tiap orang beda-beda. Ada yang cepat nangkap lewat teori, ada yang harus praktek dulu. Sayangnya, sistem pendidikan kadang belum sepenuhnya ngakomodasi perbedaan itu.

Sekarang belajar makin fleksibel. Banyak sumber gratis di internet, dari video sampai artikel. Tinggal kitanya mau atau enggak. Pendidikan sekarang lebih ke soal kemauan buat terus belajar, bukan cuma soal ijazah.

Menurutku, pendidikan yang baik itu yang bikin kita bisa berpikir kritis, bukan cuma hafal. Bisa nanya, bisa salah, dan mau belajar lagi. Selama kita masih mau belajar, berarti proses pendidikan itu masih jalan.

---TEKS 9---
TOPIK: Lingkungan

Masalah lingkungan tuh sering terasa jauh, padahal dampaknya ada di sekitar kita. Cuaca makin gak jelas, banjir di mana-mana, panas makin nyengat. Aku baru sadar kalau hal-hal kecil yang kita lakuin sehari-hari ikut berkontribusi.

Kadang kita mikir, "ah sampahku cuma sedikit". Tapi kalau semua mikir gitu, ya numpuk. Aku mulai dari hal simpel, kayak ngurangin plastik sekali pakai dan buang sampah di tempatnya. Emang kecil, tapi lebih baik daripada gak sama sekali.

Lingkungan bukan cuma tanggung jawab pemerintah, tapi kita semua. Gak harus langsung jadi aktivis. Peduli aja dulu udah langkah besar.

Kalau lingkungan rusak, yang kena dampaknya juga kita. Jadi jaga lingkungan itu sebenernya jaga masa depan sendiri. Mulai dari kebiasaan kecil, pelan-pelan tapi konsisten.

---TEKS 10---
TOPIK: Entertainment

Entertainment itu pelarian favorit banyak orang, termasuk aku. Setelah hari yang capek, nonton film, dengerin musik, atau main game rasanya kayak recharge energi. Hiburan emang punya peran penting buat jaga kewarasan.

Sekarang pilihan hiburan makin banyak. Mau nonton apa aja tinggal klik, mau denger musik genre apa aja ada. Tapi kadang kebanyakan pilihan malah bikin bingung sendiri.

Menurutku, hiburan yang baik itu yang bikin kita seneng tanpa bikin lupa waktu berlebihan. Aku pernah kebablasan binge nonton sampai begadang, ujung-ujungnya capek sendiri.

Entertainment seharusnya jadi penyeimbang hidup, bukan pelarian permanen. Selama kita bisa ngatur waktu dan gak lupa tanggung jawab, hiburan bisa jadi temen yang bikin hidup lebih ringan dan berwarna."""

# Parse texts
texts = raw_texts.split('---TEKS ')[1:]  # Skip empty first element

results = []
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

    results.append({
        'text': text_content,
        'label': 'AI',
        'source': 'GPT_Casual',
        'topic': topic,
        'word_count': word_count,
        'char_count': char_count
    })

# Save to CSV
df = pd.DataFrame(results)
output_file = 'data_ai_gpt_10.csv'
df.to_csv(output_file, index=False)

print("="*60)
print("PARSE & SAVE 10 AI TEXTS (GPT - CASUAL)")
print("="*60)
print(f"\n[OK] Saved: {output_file}")
print(f"Total: {len(df)} data\n")

for i, row in df.iterrows():
    meets = "✅" if 150 <= row['word_count'] <= 500 else "❌"
    print(f"{i+1}. {row['topic']:12} - {row['word_count']:3} kata {meets}")

print("\n" + "="*60)

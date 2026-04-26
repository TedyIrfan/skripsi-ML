# =====================================================
# ADD 10 GEMINI + 10 GPT CASUAL TEXTS
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

# Gemini texts (10)
gemini_texts = """---TEKS 1---
TOPIK: Teknologi
Jujur deh, kadang aku ngerasa ngeri sendiri sama perkembangan teknologi sekarang. Kemarin aku iseng nyoba pakai tools AI buat bantuin bikin kerangka tulisan, eh hasilnya beneran rapi banget dan logis. Rasanya kayak punya asisten pribadi yang gak pernah tidur. Tapi di sisi lain, aku jadi mikir, kalau semua hal bisa dikerjain sama robot pintar ini, terus peran kita sebagai manusia gimana, ya? Apa kita bakal jadi makin malas mikir, atau justru kita jadi bisa fokus ke hal-hal yang lebih kreatif?

Aku juga perhatiin, sekarang kita makin susah lepas dari HP. Dulu kalau lagi ngumpul sama teman, ya kita ngobrol tatap muka. Sekarang? Duduk semeja tapi matanya tertuju ke layar masing-masing. Lucu sih, kadang kita lebih connect sama orang di internet daripada orang yang duduk tepat di sebelah kita. Aku sendiri juga sering kepergok lagi scroll medsos padahal lagi diajak ngomong. Kayaknya kita perlu bikin aturan main deh, kalau lagi ngopi bareng, HP harus ditumpuk di tengah meja. Siapa yang ambil duluan, dia yang bayar bill-nya.

Tapi ya, gak bisa dipungkiri juga kalau teknologi itu ngebantu banget. Bayangin aja kalau gak ada ojek online atau maps waktu kita lagi nyasar di tempat baru. Pasti bakal repot setengah mati. Intinya sih, pintar-pintarnya kita aja ngatur waktu dan prioritas. Jangan sampai kita yang dikendaliin sama gadget, tapi kitalah yang harus tetap pegang kendali.

---TEKS 2---
TOPIK: Kesehatan
Belakangan ini aku lagi ngerasain banget dampaknya kurang tidur. Dulu pas zaman kuliah, begadang ngerjain tugas sampai subuh tuh rasanya biasa aja. Besok paginya minum kopi dikit langsung seger lagi. Tapi sekarang? Telat tidur dikit aja, besoknya badan rasanya kayak habis digebukin satu kampung. Pegal-pegal, kepala pusing, dan yang paling parah tuh mood jadi gampang banget berantakan. Kamu pernah ngalamin juga gak sih? Rasanya tuh seharian jadi gak produktif cuma gara-gara semalam keasyikan nonton film atau scroll TikTok sampai lupa waktu.

Aku jadi sadar kalau kesehatan itu investasi jangka panjang yang gak bisa ditawar. Kita sering banget ngeremehin sinyal-sinyal kecil dari tubuh, kayak sakit punggung karena kelamaan duduk atau mata perih karena layar laptop. Padahal itu cara badan kita bilang "woi, istirahat dong!". Aku sekarang lagi coba nerapin aturan simpel, kayak minum air putih minimal dua liter sehari sama jalan kaki pagi walau cuma 15 menit. Awalnya emang berat banget, apalagi ninggalin kasur pas lagi enak-enaknya.

Tapi setelah dipaksa rutin seminggu ini, kerasa sih bedanya. Badan jadi lebih enteng dan pikiran lebih fresh. Ternyata sehat itu gak harus mahal atau ribet kok. Mulai dari tidur cukup dan kurangin gula aja udah ngaruh banget. Yuk lah, kita mulai sayang sama badan sendiri sebelum badan kita yang "ngambek" minta istirahat paksa di rumah sakit.

---TEKS 3---
TOPIK: Sosial
Kalian pernah gak sih ngerasa capek banget sama tuntutan sosial buat selalu terlihat "oke"? Terutama di media sosial, rasanya semua orang tuh hidupnya sempurna banget. Ada yang baru dapet promosi kerja, ada yang liburan ke luar negeri, ada yang baru nikah. Sementara kita? Kadang lagi makan mie instan di kamar kost aja udah syukur. Perasaan FOMO atau takut ketinggalan momen ini beneran bikin capek mental. Aku sempat ada di fase di mana aku harus banget update story tiap kali lagi nongkrong di tempat hits, cuma biar dibilang gaul dan eksis.

Padahal kenyataannya, pas lagi di sana pun aku gak bener-bener nikmatin momennya karena sibuk cari angle foto yang bagus. Aneh banget kan? Kita sibuk validasi dari orang lain yang bahkan mungkin gak peduli-peduli amat sama kita. Akhir-akhir ini aku mulai belajar yang namanya JOMO (Joy of Missing Out). Ternyata enak lho sesekali gak tau apa yang lagi tren, atau gak ikutan nongkrong kalau emang badan lagi capek.

Menolak ajakan main itu bukan berarti kita jahat atau antisosial kok. Itu cuma bentuk kita menghargai energi diri sendiri. Dan ternyata, teman yang beneran teman bakal ngerti kondisi kita tanpa nge-judge. Hidup rasanya jadi lebih tenang pas kita berhenti ngebandingin "belakang panggung" kita sama "panggung utama" orang lain. Santai aja, semua orang punya timeline-nya masing-masing.

---TEKS 4---
TOPIK: Ekonomi
Ngomongin soal duit emang kadang bikin pusing, apalagi kalau lihat saldo di akhir bulan. Aku tuh sering banget kejebak sama istilah self-reward. Dikit-dikit beli kopi mahal karena merasa "ah, gue udah kerja keras hari ini". Dikit-dikit checkout belanjaan di e-commerce karena lagi diskon tanggal kembar. Padahal kalau dipikir-pikir, itu barang belum tentu butuh banget. Istilahnya tuh latte factor, pengeluaran kecil-kecil yang kalau dikumpulin ternyata bisa buat beli tiket pesawat PP ke Bali.

Sekarang aku lagi coba belajar buat lebih sadar sama pengeluaran. Bukan berarti jadi pelit sama diri sendiri ya, tapi lebih ke nahan diri buat gak impulsif. Susah banget emang, apalagi godaan paylater itu manis banget di awal tapi pahit di akhir. Kamu ngerasain juga gak sih? Pas klik bayar rasanya lega, tapi pas tagihan datang rasanya pengen pura-pura lupa ingatan.

Aku mulai nyatet pengeluaran harian di aplikasi HP, dan jujur kaget banget lihat hasilnya. Ternyata duitku paling banyak bocor di jajan makanan online. Dari situ aku mulai coba masak sendiri atau bawa bekal. Lumayan banget lho hematnya. Intinya sih, melek finansial itu penting banget buat kita anak muda. Biar nanti tua gak nyusahin orang lain dan bisa menikmati hidup dengan tenang. Yuk, mulai bijak atur dompet dari sekarang!

---TEKS 5---
TOPIK: Lingkungan
Cuaca belakangan ini bener-bener gak bisa ditebak ya? Pagi panasnya minta ampun sampai kulit rasanya kebakar, eh sorenya tiba-tiba hujan badai sampai banjir. Aku jadi kepikiran soal isu perubahan iklim yang sering dibahas orang-orang. Dulu aku mikirnya itu cuma isu elit global atau apalah, tapi sekarang kerasa banget dampaknya di kehidupan sehari-hari. Kipas angin di kamar rasanya udah gak mempan lawan panasnya siang hari.

Sebenarnya kita bisa lho bantu sedikit-sedikit buat ngerem kerusakan lingkungan ini. Aku mulai biasain bawa botol minum sendiri everywhere I go. Awalnya ribet dan sering ketinggalan, tapi lama-lama jadi kebiasaan. Selain hemat duit jajan air kemasan, kita juga ngurangin sampah plastik yang makin numpuk. Bayangin kalau satu orang buang satu botol plastik tiap hari, dikali jutaan orang, serem kan?

Hal kecil lain kayak bawa tas belanja sendiri juga ngaruh banget. Aku sering malu sih kalau ke minimarket cuma beli permen tapi dikasih kantong plastik gede. Jadi sekarang kalau belanjaannya dikit, mending aku tenteng aja atau masukin tas ransel. Kita gak perlu jadi aktivis lingkungan yang ekstrem buat peduli bumi kok. Cukup lakuin hal kecil yang konsisten, itu udah lebih dari cukup buat bantu bumi napas lega dikit.

---TEKS 6---
TOPIK: Entertainment
Kemarin aku baru aja nonton serial yang lagi hype banget di Netflix, dan sumpah itu bikin aku gak bisa tidur mikirin plot twist-nya. Seru banget ya gimana sebuah tontonan bisa mainin emosi kita sebegitunya. Dari yang ketawa ngakak, terus tiba-tiba nangis sesenggukan, sampai marah-marah sendiri ke layar TV. Kadang aku mikir, hiburan kayak film atau musik itu bukan cuma sekadar pengisi waktu luang, tapi juga obat stres paling ampuh setelah seharian kerja.

Terus fenomena konser musik sekarang juga lagi gila-gilaan banget. Perjuangan war tiket itu beneran kayak mau perang dunia. Jantung deg-degan, tangan gemeteran pas nunggu antrean online, dan pas dapat tiketnya, rasanya kayak menang lotre. Padahal harga tiketnya bisa buat makan sebulan, tapi kepuasan batin pas nyanyi bareng idola itu gak bisa dibayar pakai uang. Kamu tim yang rela nabung berbulan-bulan demi konser atau tim nunggu videonya di YouTube aja?

Dunia entertainment emang dinamis banget. Selalu ada yang baru tiap harinya. Tapi hati-hati juga, jangan sampai kita terlalu larut dalam drama dunia maya sampai lupa dunia nyata. Nikmatin aja sebagai hiburan, ambil positifnya, dan kalau udah mulai toxic atau bikin ribut sama fandom lain, mending log out dulu deh sebentar.

---TEKS 7---
TOPIK: Olahraga
Siapa di sini yang resolusi tahun barunya mau rajin olahraga tapi realitasnya wacana doang? Aku banget tuh. Rasanya tuh berat banget buat mulai gerak, apalagi kalau udah nyaman rebahan. Tapi kemarin aku dipaksa teman buat ikutan Car Free Day dan ternyata seru juga ya lari pagi bareng banyak orang. Walaupun aku lebih banyak jalan dan jajan siomay-nya daripada larinya, tapi suasananya itu lho yang bikin semangat.

Sekarang tren olahraga lari lagi booming banget. Orang-orang pada pamer pace dan jarak lari di medsos. Jujur itu agak mengintimidasi sih buat pemula kayak aku yang lari 500 meter aja udah ngos-ngosan kayak mau pingsan. Tapi aku sadar, olahraga itu bukan buat ajang pamer, tapi buat diri sendiri. Gak perlu punya sepatu mahal atau outfit branded buat mulai sehat. Cukup niat dan konsistensi aja.

Aku juga lagi coba olahraga rumahan lihat tutorial di YouTube. Modal matras doang, keringetnya udah banjir. Enaknya olahraga di rumah tuh gak ada yang ngeliatin kalau gerakan kita salah atau kaku banget kayak robot. Yang penting gerak aja dulu. Kalau badan sehat, pikiran juga jadi lebih jernih buat hadapin masalah hidup. Jadi, kapan nih kita mulai lari bareng?

---TEKS 8---
TOPIK: Budaya
Pernah gak sih kalian perhatiin budaya basa-basi di negara kita? Unik banget tapi kadang bikin keki juga. Misalnya pas lagi acara keluarga besar, pertanyaan wajibnya pasti "Kapan lulus?", "Kapan nikah?", atau "Kapan punya anak?". Padahal mungkin niat mereka cuma mau akrab atau cari topik obrolan, tapi buat yang ditanya itu rasanya kayak lagi disidang. Aku sering bingung mau jawab apa biar sopan tapi tegas.

Tapi di balik itu, budaya gotong royong dan kebersamaan kita tuh juara banget. Lihat aja kalau ada tetangga yang punya hajatan, satu kampung bisa ikut sibuk bantuin masak atau pasang tenda tanpa dibayar. Rasa kekeluargaan kayak gini yang susah ditemuin di negara lain yang lebih individualis. Walaupun kadang orang kita keponya minta ampun, tapi kepeduliannya juga tinggi banget.

Kita sebagai generasi muda punya tugas buat nyaring mana budaya yang masih relevan dan mana yang perlu disesuaikan. Gak semua tradisi lama itu kuno, dan gak semua budaya modern itu bagus. Kayak budaya antre misalnya, itu hal simpel yang harus banget kita budayakan biar lebih tertib. Yuk, kita jaga yang baik-baik dan perbaiki yang kurang pas, biar budaya kita tetap asik dan membanggakan.

---TEKS 9---
TOPIK: Pendidikan
Dulu pas sekolah, aku sering mikir, "Ngapain sih kita belajar rumus matematika yang rumit ini? Emang bakal kepakai pas beli cabe di pasar?". Tapi sekarang setelah kerja, aku baru sadar kalau pendidikan itu bukan cuma soal ngapalin materi, tapi soal pola pikir. Sekolah ngajarin kita gimana caranya memecahkan masalah, disiplin waktu, dan bersosialisasi sama orang yang beda-beda karakter.

Apalagi sekarang zaman berubah cepet banget. Ijazah doang kadang gak cukup buat bersaing. Kita dituntut buat terus belajar hal baru atau upskilling. Untungnya sekarang belajar bisa dari mana aja. Aku sering banget belajar skill baru kayak desain atau coding cuma dari nonton video tutorial gratisan. Konsep belajar seumur hidup atau long life learning itu beneran nyata dan perlu banget kita terapin.

Jangan pernah ngerasa tua buat belajar. Entah itu belajar bahasa asing baru atau belajar masak, semua ilmu pasti ada gunanya. Dan yang paling penting, jangan takut salah. Di sekolah dulu kita takut banget dapat nilai merah, tapi di dunia nyata, kegagalan itu justru guru terbaik. Jadi, mending kita jadi gelas kosong yang siap diisi ilmu baru tiap hari, biar gak ketinggalan kereta kemajuan zaman.

---TEKS 10---
TOPIK: Politik
Ngomongin politik belakangan ini rasanya kayak jalan di ladang ranjau ya? Salah ngomong dikit bisa meledak dan jadi ribut sama teman sendiri. Timeline media sosial isinya debat kusir yang gak ada ujungnya. Kubu A nyerang Kubu B, dan sebaliknya. Jujur aku kadang muak lihatnya dan milih buat mute kata kunci berbau politik biar pikiran tenang. Tapi di sisi lain, apatis atau masa bodoh juga bukan solusi yang tepat.

Kita sebagai warga negara punya hak dan tanggung jawab buat melek politik. Kebijakan yang dibuat para pejabat di sana itu ngaruh langsung ke harga beras, lapangan kerja, sampai biaya sekolah kita nanti. Jadi wajar kalau kita kritis. Cuma caranya aja mungkin yang perlu lebih santai. Debat boleh, beda pilihan boleh, tapi jangan sampai putus tali silaturahmi cuma gara-gara beda warna partai.

Penting banget buat kita cek fakta sebelum nge-share berita politik. Hoaks tuh nyebarnya cepet banget kayak virus. Kalau kita bisa jadi pemilih yang cerdas dan rasional, demokrasi kita bakal lebih sehat. Anggap aja politik itu kayak milih menu makan siang, selera orang beda-beda tapi tujuannya sama-sama mau kenyang dan sehat. Yuk, berpolitik dengan riang gembira tanpa harus saling menjatuhkan!"""

# GPT texts (10)
gpt_texts = """---TEKS 1---
TOPIK: Sosial
Aku makin ke sini makin ngerasa kalau hidup sosial itu capek tapi juga penting. Capek karena banyak drama, tapi penting karena tanpa orang lain hidup rasanya kosong. Kita tuh makhluk sosial, mau seintrovert apa pun, tetap butuh orang lain. Aku pernah ngerasain fase menarik diri, jarang ngobrol, fokus ke diri sendiri. Awalnya tenang, tapi lama-lama kok sepi banget.

Masalah sosial sekarang menurutku bukan soal gak punya temen, tapi soal susah jujur. Banyak orang tampil baik-baik aja, padahal di dalamnya lagi berantakan. Kita jadi ragu buat cerita, takut dihakimi. Akhirnya semua dipendem sendiri.

Aku belajar kalau hubungan sosial yang sehat itu gak harus rame. Cukup ada satu dua orang yang bisa diajak ngobrol jujur. Gak perlu selalu ketawa, kadang diam bareng aja udah cukup. Dari situ aku sadar, kualitas lebih penting daripada kuantitas.

Kita gak harus selalu kuat sendirian. Minta bantuan atau sekadar cerita itu bukan tanda lemah. Justru itu tanda kalau kita peduli sama diri sendiri.

---TEKS 2---
TOPIK: Budaya
Budaya itu sering dianggap sesuatu yang besar dan berat, padahal sebenernya budaya hidup di hal-hal kecil. Cara kita nyapa orang, cara makan bareng keluarga, sampai kebiasaan kumpul saat acara tertentu. Aku baru sadar itu setelah memperhatiin hal-hal sederhana di sekitar.

Kadang kita terlalu sibuk ngejar hal baru sampai lupa sama kebiasaan lama. Padahal kebiasaan itu yang bikin kita punya ciri khas. Aku pernah ngerasa kagum sama budaya luar, tapi setelah dipikir-pikir, budaya kita sendiri gak kalah menarik.

Yang bikin budaya mulai ditinggalin itu karena dianggap gak relevan. Padahal budaya bisa menyesuaikan zaman tanpa kehilangan makna. Tinggal gimana cara ngenalinnya.

Menurutku, menghargai budaya itu bukan soal harus paham semuanya, tapi soal sikap. Gak meremehkan, gak malu, dan mau belajar. Selama kita masih menghargai, budaya itu gak akan hilang.

---TEKS 3---
TOPIK: Politik
Politik sering bikin suasana panas, apalagi di media sosial. Sedikit beda pendapat, langsung ribut. Aku kadang mikir, kenapa sih orang susah banget nerima perbedaan? Padahal tujuan akhirnya sama, pengin hidup lebih baik.

Aku dulu males ngikutin politik karena ngerasa ribet. Tapi lama-lama aku sadar, politik itu gak bisa dihindari. Mau gak mau, keputusan politik ngaruh ke hidup kita sehari-hari.

Yang bikin politik jadi berat itu pas dibawa pakai emosi berlebihan. Padahal kalau dibahas santai dan pakai logika, diskusi politik bisa jadi menarik. Kita bisa beda pendapat tanpa harus saling serang.

Menurutku, yang penting itu sadar dan peduli. Gak asal ikut arus, tapi juga gak tutup mata. Politik itu soal masa depan bareng, bukan soal menang sendiri.

---TEKS 4---
TOPIK: Ekonomi
Ekonomi itu topik yang selalu bikin mikir, apalagi soal kestabilan hidup. Aku ngerasa ekonomi bukan cuma soal punya uang, tapi soal gimana kita ngelola hidup. Ada orang yang penghasilannya besar tapi tetap stres, ada juga yang sederhana tapi tenang.

Aku pernah ngerasain panik tiap akhir bulan. Bukan karena gak kerja, tapi karena pengeluaran gak terkontrol. Dari situ aku belajar, uang perlu diarahkan, bukan dibiarin jalan sendiri.

Sekarang aku lebih sadar buat nyusun prioritas. Gak semua keinginan harus diturutin. Kadang nahan diri itu berat, tapi efeknya bikin lebih tenang.

Ekonomi yang sehat itu bukan soal gaya hidup mewah, tapi soal rasa aman. Bisa tidur tanpa mikir besok makan apa, itu udah mewah banget menurutku.

---TEKS 5---
TOPIK: Teknologi
Teknologi bikin hidup kita serba cepat. Semua pengin instan, semua pengin langsung jadi. Aku ngerasa teknologi ngebantu banyak hal, tapi juga bikin kita gampang gak sabaran.

Dulu nunggu itu hal biasa, sekarang nunggu dikit aja udah kesel. Internet lemot sedikit, langsung emosi. Aku sadar, teknologi ngubah cara kita berpikir dan bereaksi.

Aku mulai nyoba buat lebih santai. Gak semua hal harus cepat. Kadang menikmati proses itu penting. Teknologi seharusnya bantu hidup, bukan bikin kita makin tertekan.

Kalau dipakai dengan sadar, teknologi bisa jadi alat yang luar biasa. Tapi kalau kebablasan, bisa bikin kita kehilangan kendali atas waktu sendiri.

---TEKS 6---
TOPIK: Kesehatan
Kesehatan mental dan fisik itu saling nyambung. Aku ngerasain sendiri, pas pikiran capek, badan ikut drop. Tapi sering kali kita cuma fokus ke fisik, lupa ke mental.

Stres sering dianggap biasa, padahal kalau dibiarin bisa numpuk. Aku belajar buat ngenalin batas diri. Kalau udah capek, ya istirahat. Gak semua harus dikejar.

Menjaga kesehatan gak harus ribet. Tidur cukup, makan lebih teratur, dan ngasih waktu buat diri sendiri itu udah langkah besar.

Sehat itu soal keseimbangan. Selama kita mau peduli dan gak memaksakan diri terus, tubuh dan pikiran bakal lebih bersahabat.

---TEKS 7---
TOPIK: Olahraga
Olahraga sering jadi hal yang niatnya gede, tapi prakteknya nol. Aku juga gitu dulu. Pengin sehat, tapi males gerak. Alasannya klasik, capek dan gak sempat.

Setelah dicoba pelan-pelan, olahraga ternyata gak semenakutkan itu. Gak harus lama, gak harus berat. Yang penting konsisten. Sedikit tapi rutin lebih kerasa daripada banyak tapi jarang.

Olahraga bikin badan lebih enak dan pikiran lebih segar. Rasanya kayak ngeluarin energi negatif yang numpuk.

Kalau olahraga udah jadi kebiasaan, efeknya kerasa ke semua aspek hidup. Tidur lebih nyenyak, mood lebih stabil, dan badan lebih siap diajak aktivitas.

---TEKS 8---
TOPIK: Pendidikan
Pendidikan sering dianggap cuma soal sekolah dan gelar. Padahal pendidikan itu proses seumur hidup. Aku ngerasa justru setelah lulus, proses belajar makin kerasa.

Banyak hal yang gak diajarin di kelas, tapi penting banget buat hidup. Cara ngatur emosi, cara komunikasi, dan cara ngambil keputusan. Semua itu kita pelajari dari pengalaman.

Sekarang belajar lebih gampang karena akses informasi luas. Tinggal kemauan kita aja. Mau berkembang atau jalan di tempat.

Pendidikan itu bukan soal siapa yang paling pintar, tapi siapa yang mau terus belajar. Selama masih mau belajar, kita gak akan tertinggal.

---TEKS 9---
TOPIK: Lingkungan
Lingkungan itu sering jadi topik besar, tapi tindakan kecilnya sering diabaikan. Aku dulu mikir, satu orang gak bakal ngaruh. Tapi setelah dipikir lagi, kalau semua mikir gitu, ya gak bakal berubah.

Aku mulai sadar dari hal sederhana, kayak lihat sampah di sekitar. Kalau dibiarkan, makin numpuk. Kalau diambil, setidaknya berkurang satu.

Jaga lingkungan itu bukan soal sempurna, tapi soal niat. Gak harus langsung ekstrem, cukup konsisten. Hal kecil yang dilakukan banyak orang bisa jadi perubahan besar.

Lingkungan yang sehat bikin hidup kita juga lebih nyaman. Jadi peduli lingkungan itu sebenernya peduli sama diri sendiri.

---TEKS 10---
TOPIK: Entertainment
Entertainment itu bagian dari hidup yang gak bisa dipisahin. Tanpa hiburan, hidup rasanya kering. Aku sering pakai hiburan buat ngilangin penat setelah hari yang panjang.

Tapi hiburan juga bisa jadi jebakan kalau gak diatur. Niatnya santai, malah kebablasan. Aku pernah nonton atau main sampai lupa waktu, dan akhirnya capek sendiri.

Sekarang aku coba lebih sadar. Hiburan secukupnya, tanggung jawab tetap jalan. Kalau seimbang, hiburan bisa jadi sumber energi, bukan pelarian.

Entertainment yang sehat itu yang bikin kita balik ke dunia nyata dengan perasaan lebih baik."""

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

# Parse both batches
gemini_data = parse_texts(gemini_texts, 'Gemini_Casual')
gpt_data = parse_texts(gpt_texts, 'GPT_Casual')

# Add all to existing data
existing_data.extend(gemini_data)
existing_data.extend(gpt_data)

# Save all data
df = pd.DataFrame(existing_data)
df.to_csv(OUTPUT_FILE, index=False)

print("="*60)
print("ADD 10 GEMINI + 10 GPT CASUAL TEXTS")
print("="*60)
print(f"\n[OK] Saved: {OUTPUT_FILE}")
print(f"Total: {len(df)} data\n")

for i, row in df.iterrows():
    print(f"{i+1}. {row['source']:12} - {row['topic']:12} - {row['word_count']:3} kata")

print("\n" + "="*60)

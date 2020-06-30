## BERT NLP

This project using huggingface library for test the model.

Tujuan dari project ini adalah untuk melakukan klasifikasi pada sejumlah teks.


BERT paper: https://arxiv.org/pdf/1810.04805.pdf

[Bert base uncase](https://huggingface.co/bert-base-uncased#:~:text=BERT%20base%20model%20(uncased),difference%20between%20english%20and%20English.) : adalah sebuah metode yang diperkenalkan dipaper bert dimana memberitahukan pada model bahwa tidak ada perbedaan huruf kapital dan huruf biasa.
Contoh: "makan" dan "Makan", kedua kata ini adalah "sama" atau tidak ada perbedaan.

Step:
1. Mengubah kalimat menjadi `tokenized`
2. Mengubah `tokenized` menjadi `Token IDs`
3. Setiap kalimat/paragraf yang diubah menjadi `Token IDs` akan ditambahin dengan `Special Token`
4. `Special Token` awal kalimat/paragram disebut `[SEP]` dan akhir kalimat/paragraf disebut `[CLS]`
5. Bert harus memiliki maksimum sentence length artinya kalimat yang paling panjang maka akan menjadi `Maximum sentence length` dan jika kalimat yang lebih pendek maka sisanya akan diisi dengan `PAD`. Sedangkan kode pada [baris 98~125](https://github.com/akyong/BERT-NLP-TEST/blob/master/BERT.py#L98-L125) sudah meliputi langkah 1 hingga 4 diatas.

`Tokenized`: kalimat yang dipisahin perkata termasuk koma, petik dan titik.
`Token IDs`: kata yang diubah menjadi sebuah token dalam bentuk angka. e.g: 2556

Saat melakukan training menggunakan `pytorch` dimana menggunakan 90% dataset dan 10% digunakan sebagai data loader.


Question: 
1. Kenapa dataset harus 90% digunakan untuk training dan 10% digunakan sebagai validasi?

Author: Bobby Akyong

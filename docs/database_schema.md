# Database Schema Information

## BPJS Antrol Tables

### reg_periksa
- no_rawat: Registration number
- tgl_registrasi: Registration date
- jam_reg: Registration time
- kd_dokter: Doctor code
- no_rkm_medis: Medical record number
- kd_poli: Polyclinic code
- status_lanjut: Continuation status
- kd_pj: Payment method code

### mlite_antrian_referensi
- tanggal_periksa: Examination date
- nomor_kartu: Card number
- nomor_referensi: Reference number
- kodebooking: Booking code
- jenis_kunjungan: Visit type
- status_kirim: Send status
- keterangan: Notes

### Other related tables
- poliklinik: Polyclinic information
- dokter: Doctor information
- penjab: Payment method information
- pasien: Patient information
- bridging_sep: SEP bridging information

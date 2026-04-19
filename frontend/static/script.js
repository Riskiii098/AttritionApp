document.addEventListener('DOMContentLoaded', function() {
    const predictForm = document.getElementById('predictForm');
    
    if (predictForm) {
        predictForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const resultBox = document.getElementById('resultBox');
            const resultText = document.getElementById('resultText');
            const submitBtn = document.getElementById('submitBtn');

            // Tangkap Element Value
            const incomeVal = document.getElementById('inputSalary').value;
            const ageVal = document.getElementById('inputAge').value;
            const distanceVal = document.getElementById('inputDistance').value;
            const overtimeVal = document.getElementById('inputOverTime').value;
            const totalYearsVal = document.getElementById('inputTotalYears').value;
            const companyYearsVal = document.getElementById('inputYearsCompany').value;
            const numCompaniesVal = document.getElementById('inputNumCompanies').value;
            const jobSatisfVal = document.getElementById('inputJobSatisf').value;

            // Simpan state tombol
            const originalBtnText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Memvalidasi & Berpikir...';
            submitBtn.disabled = true;
            resultBox.style.display = 'none';

            try {
                // Parsing nilai mentah ke format Number
                const income = parseInt(incomeVal);
                const age = parseInt(ageVal);
                const distance = parseInt(distanceVal);
                const totalYears = parseInt(totalYearsVal);
                const companyYears = parseInt(companyYearsVal);
                const numCompanies = parseInt(numCompaniesVal);
                const jobSatisf = parseInt(jobSatisfVal);

                // --- SISTEM VALIDASI SILANG MASUK AKAL (LOGICAL CONSTRAINTS) ---
                
                // 1. Validasi Masa Kerja vs Lama di Perusahaan
                if (companyYears > totalYears) {
                    throw new Error(`<b>Logika Salah:</b> Karyawan tidak mungkin bekerja di perusahaan ini selama <b>${companyYears} tahun</b>, sementara total pengalaman kerjanya seumur hidup saja baru <b>${totalYears} tahun</b>.`);
                }

                // 2. Validasi Usia vs Total Pengalaman Kerja (Asumsi mulai bekerja legal tercepat usia 15 tahun)
                const maxLogicYears = age - 15;
                if (totalYears > maxLogicYears) {
                    let errMsg = `<b>Logika Usia Salah:</b> Jika karyawan berusia ${age} tahun, sungguh tidak masuk akal memiliki total masa kerja ${totalYears} tahun!`;
                    if (maxLogicYears <= 0) {
                        errMsg += ` (Dia bahkan belum cukup umur / masih balita saat mulai).`;
                    } else {
                        errMsg += ` (Maksimal masa karir yang rasional untuk Usia ${age} adalah ${maxLogicYears} tahun pengalaman).`;
                    }
                    throw new Error(errMsg);
                }

                // 3. Validasi Rentang Ekstrem yang merusak struktur tree Scikit-Learn
                if (income > 50000 || income < 500) {
                    throw new Error(`<b>Gaji Out of Bounds:</b> Atribut model latih hanya mengenali profil rata-rata dengan batas rasional 500 USD hingga 50.000 USD.`);
                }
                
                if (numCompanies > totalYears && totalYears > 0) {
                    throw new Error(`<b>Pindah-Pindah Ekstrem:</b> Memiliki pengalaman ${numCompanies} perusahaan namun total masa kerja baru ${totalYears} tahun sangat tidak relevan untuk standar pengukuran HR.`);
                }


                // Lakukan mapping jika semua logika aman
                const userPayloadObject = {
                    "MonthlyIncome": income,
                    "Age": age,
                    "DistanceFromHome": distance,
                    "OverTime": overtimeVal,
                    "TotalWorkingYears": totalYears,
                    "YearsAtCompany": companyYears,
                    "NumCompaniesWorked": numCompanies,
                    "JobSatisfaction": jobSatisf
                };

                // Request API REST Flask
                // Gunakan jalur relatif tanpa '/' di depan agar lebih aman di proxy Hugging Face
                const response = await fetch('api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ features: userPayloadObject })
                });

                // Cek jika respons bukan JSON
                const contentType = response.headers.get("content-type");
                if (!contentType || !contentType.includes("application/json")) {
                    const errorHtml = await response.text();
                    console.error("Respons bukan JSON:", errorHtml);
                    throw new Error(`Server tidak mengembalikan JSON. Status: ${response.status}. Hubungi admin atau cek log server.`);
                }

                const data = await response.json();
                resultBox.style.display = 'block';

                if (response.ok && data.status === 'success') {
                    if (data.prediction === 1) { // 1 = Attrition
                        resultBox.className = 'result-box mt-4 alert alert-danger border-0 shadow-lg';
                        resultText.innerHTML = '⚠️ <strong>RISIKO RESIGN TINGGI</strong><br><span class="fs-5 text-dark">Model menyimpulkan Profil Karyawan ini rentan untuk keluar perusahaan dalam waktu dekat.<br>Tingkat Probabilitas Resign: <br><strong><span class="fs-1 text-danger">' + data.probability_percentage + '%</span></strong>.</span>';
                    } else { // 0 = Loyal
                        resultBox.className = 'result-box mt-4 alert alert-success border-0 shadow-sm';
                        resultText.innerHTML = '✅ <strong>KATEGORI AMAN / MERASAKAN NYAMAN</strong><br><span class="fs-5 text-dark">Karyawan lebih cenderung Memilih Bertahan dan Setia. Indikasi keinginan keluar hanyalah:<br><strong><span class="fs-1 text-success">' + data.probability_percentage + '%</span></strong>.</span>';
                    }
                } else {
                    resultBox.className = 'result-box mt-4 alert alert-warning border-0';
                    resultText.innerHTML = `<strong>Error Internal API:</strong> ${data.message || 'Respons API bermasalah.'}`;
                }

            } catch (error) {
                // Tampilkan Error Peringatan Form Tidak Masuk Akal
                resultBox.style.display = 'block';
                resultBox.className = 'result-box mt-4 alert alert-warning border-0 shadow-sm text-start';
                resultText.innerHTML = `<span class="text-danger fw-bolder fs-5">Form Cacat Logika (Ditolak)</span><hr class="mt-1 mb-2 border-danger opacity-25"> <span class="text-dark fs-6">${error.message}</span><br><br><small class="text-muted fw-normal">Silahkan perbaiki nilai form Anda terlebih dahulu agar AI dapat memproses probabilitas yang akurat berdasarkan realita.</small>`;
                
            } finally {
                // Matikan mode loading (kembali ke state awal)
                submitBtn.innerHTML = originalBtnText;
                submitBtn.disabled = false;
            }
        });
    }
});

# Performans Optimizasyon Önerileri

Bu belge, Qwen3 projesinin performansını, özellikle yüksek eş zamanlılık ve büyük veri setlerinin işlenmesi senaryolarında iyileştirmek için önerilen teknikleri özetlemektedir.

## 1. API Sunucusu (`qwen3-api.py`) Optimizasyonları

API sunucusu, sistemin ana giriş noktasıdır ve buradaki verimlilik genel performansı doğrudan etkiler.

### 1.1. Asenkron (Asynchronous) Mimarisine Geçiş

- **Mevcut Durum:** `requests` kütüphanesi senkron çalıştığı için, API sunucusu Ollama'dan bir embedding yanıtı beklerken diğer gelen istekleri işleyemez (bloke olur).
- **Öneri:** `requests` kütüphanesini `httpx` ile değiştirmek ve tüm embedding oluşturma sürecini `async` ve `await` kullanarak asenkron hale getirmek.
- **Fayda:** Sunucu, bir isteğin tamamlanmasını beklerken diğer istekleri de kabul edip işlemeye başlayabilir. Bu, eş zamanlı kullanıcı kapasitesini ve genel yanıt verebilirliği (responsiveness) önemli ölçüde artırır.

### 1.2. Toplu İsteklerin (Batch Requests) Paralel İşlenmesi

- **Mevcut Durum:** API'ye bir liste halinde birden çok metin gönderildiğinde (`["metin1", "metin2", ...]`), her bir metin için Ollama'ya sırayla ve tek tek istek atılır.
- **Öneri:** Gelen listedeki tüm metinler için embedding oluşturma isteklerini `asyncio.gather` kullanarak Ollama'ya aynı anda (paralel olarak) göndermek.
- **Fayda:** 10 metinlik bir istek, 10 ayrı istek süresi yerine neredeyse tek bir istek süresinde tamamlanır. Bu, özellikle toplu veri işleme (batch processing) senaryolarında performansı katbekat artırır.

### 1.3. Daha Hızlı Bellek-içi (In-Memory) Önbellekleme

- **Mevcut Durum:** Önbellekleme (caching) disk üzerindeki dosyalara yapılıyor. Disk G/Ç (okuma/yazma) işlemleri, bellek işlemlerine göre yavaştır.
- **Öneri:** `cachetools` gibi bir kütüphane kullanarak, sık erişilen embedding'leri doğrudan bellekte (RAM) tutan akıllı bir önbellek mekanizması (örneğin, TTL - Time-to-Live veya LFU - Least Frequently Used) kurmak.
- **Fayda:** Tekrarlanan istekler için yanıt süresi, disk yerine çok daha hızlı olan RAM'den okuma yapıldığı için milisaniyeler seviyesine düşer.

## 2. Veri İndeksleme (`qdrantsetup.py`) Optimizasyonları

Büyük hacimli verileri Qdrant'a eklerken en büyük darboğaz, her bir belge için API'ye ayrı ayrı HTTP istekleri yapmaktır.

### 2.1. Toplu Embedding Oluşturma (Batch Embedding)

- **Mevcut Durum:** `qdrantsetup.py` betiğindeki `add_documents_batch` fonksiyonu, belgeleri Qdrant'a toplu olarak *eklese de*, her bir belgenin embedding'ini API'den *tek tek* ister. Bu, yüzlerce veya binlerce yavaş HTTP isteği anlamına gelir.
- **Öneri:** `qdrantsetup.py` betiğini, belirli bir boyuttaki (örneğin 100) belge metnini tek bir listede toplayıp, API'nin `/v1/embeddings` endpoint'ine tek bir toplu istekte gönderecek şekilde güncellemek. API tarafında yapılacak paralel işleme optimizasyonu sayesinde, bu istek çok verimli bir şekilde karşılanacaktır.
- **Fayda:** Yüzlerce belgeyi indekslemek için yüzlerce HTTP isteği yapmak yerine sadece birkaç HTTP isteği yapılır. Bu, büyük veri setlerinin indeksleme süresini **önemli ölçüde** kısaltır.

## Uygulama Önceliği

| Öneri | Etki Alanı | Getireceği Performans Kazancı | Uygulama Önceliği |
| :--- | :--- | :--- | :--- |
| Asenkron API & Paralel İşleme | API Sunucusu | **Çok Yüksek** | **Yüksek** |
| Toplu Embedding Alımı | Veri İndeksleme | **Çok Yüksek** | **Yüksek** |
| Bellek-içi Önbellek | API Sunucusu | **Orta** | Orta |

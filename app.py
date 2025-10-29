import streamlit as st
import pandas as pd
import datetime
import json
import os
from datetime import datetime, date

# Sayfa yapılandırması
st.set_page_config(
    page_title="Öğrenci-Veli Portalı",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Veri dosyası yolu
DATA_FILE = "ogrenci_verileri.json"

# Başlangıç verilerini yükleme veya oluşturma
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # Örnek başlangıç verileri
        default_data = {
            "duyurular": [
                {
                    "id": 1,
                    "baslik": "Hoş Geldiniz",
                    "icerik": "Öğrenci-Veli portalımıza hoş geldiniz. Buradan duyuruları, ödevleri ve değerlendirmeleri takip edebilirsiniz.",
                    "tarih": "2023-09-01",
                    "onem": "Yüksek"
                }
            ],
            "odevler": [
                {
                    "id": 1,
                    "ders": "Matematik",
                    "konu": "Kesirler",
                    "aciklama": "Sayfa 45-46 arasındaki alıştırmalar yapılacak.",
                    "teslim_tarihi": "2023-09-10",
                    "durum": "Bekliyor"
                }
            ],
            "degerlendirmeler": [
                {
                    "id": 1,
                    "ders": "Matematik",
                    "sinav_adi": "1. Değerlendirme Sınavı",
                    "puan": 85,
                    "tarih": "2023-09-05",
                    "aciklama": "Kesirler konusunda başarılı."
                }
            ],
            "etkinlikler": [
                {
                    "id": 1,
                    "etkinlik_adi": "Okul Gezisi",
                    "yer": "Bilim Müzesi",
                    "tarih": "2023-09-15",
                    "aciklama": "Bilim müzesine gezi düzenlenecektir."
                }
            ],
            "devamsizlik": [
                {
                    "id": 1,
                    "tarih": "2023-09-03",
                    "ders": "Matematik",
                    "sebep": "Hastalık",
                    "durum": "Onaylandı"
                }
            ]
        }
        save_data(default_data)
        return default_data

# Verileri kaydetme
def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Ana başlık
st.title("📚 Öğrenci-Veli Portalı")
st.markdown("---")

# Yan menü
menu = st.sidebar.selectbox(
    "Menü",
    ["Ana Sayfa", "Duyurular", "Ödev Takibi", "Değerlendirmeler", "Etkinlik Takvimi", "Devamsızlık Bilgisi", "İletişim"]
)

# Verileri yükle
data = load_data()

# Ana Sayfa
if menu == "Ana Sayfa":
    st.header("Hoş Geldiniz!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📢 Son Duyurular")
        for duyuru in data["duyurular"][-3:]:
            st.info(f"**{duyuru['baslik']}** - {duyuru['tarih']}")
    
    with col2:
        st.subheader("📝 Yaklaşan Ödevler")
        bugun = date.today()
        for odev in data["odevler"]:
            if odev["durum"] == "Bekliyor":
                teslim_tarihi = datetime.strptime(odev["teslim_tarihi"], "%Y-%m-%d").date()
                if teslim_tarihi >= bugun:
                    st.warning(f"**{odev['ders']}** - {odev['konu']}")
    
    with col3:
        st.subheader("📊 Son Değerlendirmeler")
        for degerlendirme in data["degerlendirmeler"][-3:]:
            st.success(f"**{degerlendirme['ders']}** - {degerlendirme['puan']}")

# Duyurular Sayfası
elif menu == "Duyurular":
    st.header("📢 Duyurular")
    
    # Yeni duyuru ekleme
    with st.expander("Yeni Duyuru Ekle"):
        with st.form("duyuru_ekle"):
            baslik = st.text_input("Duyuru Başlığı")
            icerik = st.text_area("Duyuru İçeriği")
            onem = st.selectbox("Önem Derecesi", ["Düşük", "Orta", "Yüksek"])
            tarih = st.date_input("Tarih")
            
            if st.form_submit_button("Duyuruyu Ekle"):
                if baslik and icerik:
                    yeni_duyuru = {
                        "id": max([d["id"] for d in data["duyurular"]]) + 1 if data["duyurular"] else 1,
                        "baslik": baslik,
                        "icerik": icerik,
                        "onem": onem,
                        "tarih": tarih.strftime("%Y-%m-%d")
                    }
                    data["duyurular"].append(yeni_duyuru)
                    save_data(data)
                    st.success("Duyuru başarıyla eklendi!")
                    st.rerun()
    
    # Duyuruları listeleme
    for duyuru in reversed(data["duyurular"]):
        if duyuru["onem"] == "Yüksek":
            st.error(f"**{duyuru['baslik']}** - {duyuru['tarih']}")
        elif duyuru["onem"] == "Orta":
            st.warning(f"**{duyuru['baslik']}** - {duyuru['tarih']}")
        else:
            st.info(f"**{duyuru['baslik']}** - {duyuru['tarih']}")
        
        with st.expander("Detayları Gör"):
            st.write(duyuru["icerik"])
            
            if st.button("Sil", key=f"sil_duyuru_{duyuru['id']}"):
                data["duyurular"] = [d for d in data["duyurular"] if d["id"] != duyuru["id"]]
                save_data(data)
                st.rerun()

# Ödev Takibi Sayfası
elif menu == "Ödev Takibi":
    st.header("📝 Ödev Takibi")
    
    # Yeni ödev ekleme
    with st.expander("Yeni Ödev Ekle"):
        with st.form("odev_ekle"):
            ders = st.text_input("Ders Adı")
            konu = st.text_input("Ödev Konusu")
            aciklama = st.text_area("Açıklama")
            teslim_tarihi = st.date_input("Teslim Tarihi")
            durum = st.selectbox("Durum", ["Bekliyor", "Tamamlandı", "Ertelendi"])
            
            if st.form_submit_button("Ödevi Ekle"):
                if ders and konu:
                    yeni_odev = {
                        "id": max([o["id"] for o in data["odevler"]]) + 1 if data["odevler"] else 1,
                        "ders": ders,
                        "konu": konu,
                        "aciklama": aciklama,
                        "teslim_tarihi": teslim_tarihi.strftime("%Y-%m-%d"),
                        "durum": durum
                    }
                    data["odevler"].append(yeni_odev)
                    save_data(data)
                    st.success("Ödev başarıyla eklendi!")
                    st.rerun()
    
    # Ödevleri listeleme
    for odev in data["odevler"]:
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.subheader(f"{odev['ders']} - {odev['konu']}")
            st.write(odev['aciklama'])
        
        with col2:
            teslim_tarihi = datetime.strptime(odev["teslim_tarihi"], "%Y-%m-%d").date()
            bugun = date.today()
            if teslim_tarihi < bugun and odev["durum"] != "Tamamlandı":
                st.error(f"Teslim Tarihi: {odev['teslim_tarihi']} (Geçmiş!)")
            else:
                st.write(f"Teslim Tarihi: {odev['teslim_tarihi']}")
            
            if odev["durum"] == "Tamamlandı":
                st.success(odev["durum"])
            elif odev["durum"] == "Ertelendi":
                st.warning(odev["durum"])
            else:
                st.info(odev["durum"])
        
        with col3:
            if st.button("Durumu Güncelle", key=f"guncelle_{odev['id']}"):
                # Durum güncelleme mantığı buraya eklenebilir
                st.info("Durum güncelleme özelliği aktif edilecek")
            
            if st.button("Sil", key=f"sil_odev_{odev['id']}"):
                data["odevler"] = [o for o in data["odevler"] if o["id"] != odev["id"]]
                save_data(data)
                st.rerun()
        
        st.markdown("---")

# Değerlendirmeler Sayfası
elif menu == "Değerlendirmeler":
    st.header("📊 Değerlendirmeler")
    
    # Yeni değerlendirme ekleme
    with st.expander("Yeni Değerlendirme Ekle"):
        with st.form("degerlendirme_ekle"):
            ders = st.text_input("Ders Adı")
            sinav_adi = st.text_input("Sınav Adı")
            puan = st.slider("Puan", 0, 100, 50)
            tarih = st.date_input("Tarih")
            aciklama = st.text_area("Açıklama")
            
            if st.form_submit_button("Değerlendirmeyi Ekle"):
                if ders and sinav_adi:
                    yeni_degerlendirme = {
                        "id": max([d["id"] for d in data["degerlendirmeler"]]) + 1 if data["degerlendirmeler"] else 1,
                        "ders": ders,
                        "sinav_adi": sinav_adi,
                        "puan": puan,
                        "tarih": tarih.strftime("%Y-%m-%d"),
                        "aciklama": aciklama
                    }
                    data["degerlendirmeler"].append(yeni_degerlendirme)
                    save_data(data)
                    st.success("Değerlendirme başarıyla eklendi!")
                    st.rerun()
    
    # Değerlendirmeleri listeleme
    for degerlendirme in data["degerlendirmeler"]:
        col1, col2, col3 = st.columns([3, 1, 2])
        
        with col1:
            st.subheader(f"{degerlendirme['ders']} - {degerlendirme['sinav_adi']}")
            st.write(f"Tarih: {degerlendirme['tarih']}")
            st.write(degerlendirme['aciklama'])
        
        with col2:
            # Puanı renkli gösterme
            puan = degerlendirme['puan']
            if puan >= 85:
                st.success(f"**{puan}**")
            elif puan >= 70:
                st.warning(f"**{puan}**")
            else:
                st.error(f"**{puan}**")
        
        with col3:
            if st.button("Sil", key=f"sil_degerlendirme_{degerlendirme['id']}"):
                data["degerlendirmeler"] = [d for d in data["degerlendirmeler"] if d["id"] != degerlendirme["id"]]
                save_data(data)
                st.rerun()
        
        st.markdown("---")
    
    # Not ortalaması hesaplama
    if data["degerlendirmeler"]:
        ortalama = sum([d["puan"] for d in data["degerlendirmeler"]]) / len(data["degerlendirmeler"])
        st.metric("Genel Not Ortalaması", f"{ortalama:.2f}")

# Etkinlik Takvimi Sayfası
elif menu == "Etkinlik Takvimi":
    st.header("📅 Etkinlik Takvimi")
    
    # Yeni etkinlik ekleme
    with st.expander("Yeni Etkinlik Ekle"):
        with st.form("etkinlik_ekle"):
            etkinlik_adi = st.text_input("Etkinlik Adı")
            yer = st.text_input("Yer")
            tarih = st.date_input("Tarih")
            aciklama = st.text_area("Açıklama")
            
            if st.form_submit_button("Etkinliği Ekle"):
                if etkinlik_adi and yer:
                    yeni_etkinlik = {
                        "id": max([e["id"] for e in data["etkinlikler"]]) + 1 if data["etkinlikler"] else 1,
                        "etkinlik_adi": etkinlik_adi,
                        "yer": yer,
                        "tarih": tarih.strftime("%Y-%m-%d"),
                        "aciklama": aciklama
                    }
                    data["etkinlikler"].append(yeni_etkinlik)
                    save_data(data)
                    st.success("Etkinlik başarıyla eklendi!")
                    st.rerun()
    
    # Etkinlikleri listeleme
    for etkinlik in data["etkinlikler"]:
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.subheader(etkinlik['etkinlik_adi'])
            st.write(f"Yer: {etkinlik['yer']}")
            st.write(etkinlik['aciklama'])
        
        with col2:
            st.write(f"Tarih: {etkinlik['tarih']}")
        
        with col3:
            if st.button("Sil", key=f"sil_etkinlik_{etkinlik['id']}"):
                data["etkinlikler"] = [e for e in data["etkinlikler"] if e["id"] != etkinlik["id"]]
                save_data(data)
                st.rerun()
        
        st.markdown("---")

# Devamsızlık Bilgisi Sayfası
elif menu == "Devamsızlık Bilgisi":
    st.header("📋 Devamsızlık Bilgisi")
    
    # Yeni devamsızlık ekleme
    with st.expander("Yeni Devamsızlık Ekle"):
        with st.form("devamsizlik_ekle"):
            tarih = st.date_input("Tarih")
            ders = st.text_input("Ders Adı")
            sebep = st.text_input("Sebep")
            durum = st.selectbox("Durum", ["Onaylandı", "Beklemede", "Reddedildi"])
            
            if st.form_submit_button("Devamsızlığı Ekle"):
                if ders and sebep:
                    yeni_devamsizlik = {
                        "id": max([d["id"] for d in data["devamsizlik"]]) + 1 if data["devamsizlik"] else 1,
                        "tarih": tarih.strftime("%Y-%m-%d"),
                        "ders": ders,
                        "sebep": sebep,
                        "durum": durum
                    }
                    data["devamsizlik"].append(yeni_devamsizlik)
                    save_data(data)
                    st.success("Devamsızlık başarıyla eklendi!")
                    st.rerun()
    
    # Devamsızlıkları listeleme
    for devamsizlik in data["devamsizlik"]:
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.write(f"**Tarih:** {devamsizlik['tarih']}")
            st.write(f"**Ders:** {devamsizlik['ders']}")
        
        with col2:
            st.write(f"**Sebep:** {devamsizlik['sebep']}")
            
            if devamsizlik["durum"] == "Onaylandı":
                st.success(devamsizlik["durum"])
            elif devamsizlik["durum"] == "Beklemede":
                st.warning(devamsizlik["durum"])
            else:
                st.error(devamsizlik["durum"])
        
        with col3:
            if st.button("Sil", key=f"sil_devamsizlik_{devamsizlik['id']}"):
                data["devamsizlik"] = [d for d in data["devamsizlik"] if d["id"] != devamsizlik["id"]]
                save_data(data)
                st.rerun()
        
        st.markdown("---")
    
    # Devamsızlık istatistikleri
    if data["devamsizlik"]:
        toplam_devamsizlik = len(data["devamsizlik"])
        onaylanan_devamsizlik = len([d for d in data["devamsizlik"] if d["durum"] == "Onaylandı"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Toplam Devamsızlık", toplam_devamsizlik)
        with col2:
            st.metric("Onaylanan Devamsızlık", onaylanan_devamsizlik)

# İletişim Sayfası
elif menu == "İletişim":
    st.header("📞 İletişim")
    
    st.subheader("Okul Bilgileri")
    st.write("**Adres:** Örnek Mahallesi, Okul Caddesi No:123, İstanbul")
    st.write("**Telefon:** (0212) 123 45 67")
    st.write("**E-posta:** info@ornekokul.com")
    
    st.subheader("Öğretmenler")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sınıf Öğretmeni:** Ayşe Yılmaz")
        st.write("**E-posta:** ayse.yilmaz@ornekokul.com")
        st.write("**Telefon:** (0212) 123 45 68")
    
    with col2:
        st.write("**Müdür:** Mehmet Demir")
        st.write("**E-posta:** mehmet.demir@ornekokul.com")
        st.write("**Telefon:** (0212) 123 45 69")
    
    st.subheader("Mesaj Gönder")
    with st.form("mesaj_formu"):
        ad_soyad = st.text_input("Adınız Soyadınız")
        eposta = st.text_input("E-posta Adresiniz")
        konu = st.selectbox("Konu", ["Genel", "Ödevler", "Değerlendirmeler", "Devamsızlık", "Diğer"])
        mesaj = st.text_area("Mesajınız")
        
        if st.form_submit_button("Mesajı Gönder"):
            if ad_soyad and eposta and mesaj:
                st.success("Mesajınız başarıyla gönderildi! En kısa sürede size dönüş yapılacaktır.")
            else:
                st.error("Lütfen tüm alanları doldurun.")

# Alt bilgi
st.markdown("---")
st.markdown("© 2023 Örnek Okul - Öğrenci Veli Portalı")

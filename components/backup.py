"""Backup, archive and restore functionality."""
import streamlit as st
import json
import io
import csv
from datetime import datetime
from utils.data_manager import save_data, DATA_FILE


# ── CSV column order ──────────────────────────────────────────────────────────
_TRADE_COLS = [
    "id", "symbol", "direction", "result", "pnl", "r_multiple",
    "capital", "risk_pct", "avg_entry", "stop_loss", "position_size",
    "setup_type", "market_condition", "emotion", "execution_score",
    "plan_followed", "mistakes", "comment", "rr_display",
    "created_at", "closed_at", "notes",
]


def _trades_to_csv(trades: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_TRADE_COLS, extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    for t in trades:
        row = {k: t.get(k, "") for k in _TRADE_COLS}
        # Flatten list fields
        if isinstance(row.get("mistakes"), list):
            row["mistakes"] = "; ".join(row["mistakes"])
        writer.writerow(row)
    return buf.getvalue()


def _now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def render_backup(data: dict):
    closed  = data.get("closed_trades", [])
    active  = data.get("active_positions", [])

    st.markdown('<div class="section-title">💾 Yedekleme & Arşiv</div>', unsafe_allow_html=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="card" style="display:flex;gap:2rem;flex-wrap:wrap">
      <div>
        <div class="detail-label">Kapalı İşlem</div>
        <div class="stat-value">{len(closed)}</div>
      </div>
      <div>
        <div class="detail-label">Aktif Pozisyon</div>
        <div class="stat-value">{len(active)}</div>
      </div>
      <div>
        <div class="detail-label">Toplam Kayıt</div>
        <div class="stat-value">{len(closed) + len(active)}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Export section ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📤 Dışa Aktar</div>', unsafe_allow_html=True)

    ec1, ec2, ec3 = st.columns(3)

    # JSON full backup
    with ec1:
        st.markdown("**JSON Tam Yedek**")
        st.caption("Tüm veriyi (aktif + kapalı) içerir. Geri yüklemek için kullanın.")
        json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            label="⬇️ JSON indir",
            data=json_bytes,
            file_name=f"tradевault_backup_{_now_str()}.json",
            mime="application/json",
            use_container_width=True,
            key="dl_json_full",
        )

    # CSV closed trades
    with ec2:
        st.markdown("**CSV — Kapalı İşlemler**")
        st.caption("Excel ile açılabilir. Analiz için idealdir.")
        if closed:
            csv_data = _trades_to_csv(closed)
            st.download_button(
                label="⬇️ CSV indir",
                data=csv_data.encode("utf-8-sig"),   # utf-8-sig = Excel BOM
                file_name=f"closed_trades_{_now_str()}.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_csv_closed",
            )
        else:
            st.info("Kapalı işlem yok.")

    # CSV active positions
    with ec3:
        st.markdown("**CSV — Aktif Pozisyonlar**")
        st.caption("Aktif pozisyon listesi.")
        if active:
            csv_data_a = _trades_to_csv(active)
            st.download_button(
                label="⬇️ CSV indir",
                data=csv_data_a.encode("utf-8-sig"),
                file_name=f"active_positions_{_now_str()}.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_csv_active",
            )
        else:
            st.info("Aktif pozisyon yok.")

    # ── Archive (move closed to separate store) ────────────────────────────────
    st.markdown('<div class="section-title">🗄️ Arşivle</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card" style="color:#8b949e;font-size:0.85rem;line-height:1.6">
      Arşivleme: seçilen kapalı işlemleri ana listeden kaldırır ve ayrı bir 
      <code>trades_archive_TARIH.json</code> dosyasına yazar. 
      Ana veriler korunur.
    </div>
    """, unsafe_allow_html=True)

    if closed:
        arch_col1, arch_col2 = st.columns([2, 3])
        with arch_col1:
            archive_all = st.checkbox("Tüm kapalı işlemleri arşivle", key="arch_all")
        with arch_col2:
            arch_symbols = []
            if not archive_all:
                symbols = sorted(set(t.get("symbol", "") for t in closed))
                arch_symbols = st.multiselect("Sembol seç", symbols, key="arch_symbols")

        if st.button("🗄️ Arşivle ve Listeden Kaldır", key="do_archive", type="primary"):
            if archive_all or not arch_symbols:
                to_archive = closed[:]
                keep       = []
            else:
                to_archive = [t for t in closed if t.get("symbol") in arch_symbols]
                keep       = [t for t in closed if t.get("symbol") not in arch_symbols]

            if to_archive:
                # Build archive file content
                archive_payload = {
                    "archived_at": datetime.now().isoformat(),
                    "trades":      to_archive,
                }
                archive_json = json.dumps(archive_payload, ensure_ascii=False, indent=2).encode("utf-8")

                # Offer download
                st.download_button(
                    label=f"⬇️ Arşiv dosyasını indir ({len(to_archive)} işlem)",
                    data=archive_json,
                    file_name=f"archive_{_now_str()}.json",
                    mime="application/json",
                    key="dl_archive",
                )

                # Remove from live data
                data["closed_trades"] = keep
                save_data(data)
                st.session_state.data = data
                st.success(f"✅ {len(to_archive)} işlem arşivlendi ve listeden kaldırıldı.")
                st.rerun()
            else:
                st.warning("Arşivlenecek işlem bulunamadı.")
    else:
        st.info("Arşivlenecek kapalı işlem yok.")

    # ── Restore / Import ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📥 Geri Yükle / İçe Aktar</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="color:#8b949e;font-size:0.85rem;line-height:1.6">
      <b style="color:#c9d1d9">JSON tam yedek</b> yükleyerek tüm veriyi geri yükleyebilirsiniz.<br>
      <b style="color:#e3b341">⚠️ Dikkat:</b> Bu işlem mevcut veriyi tamamen değiştirir.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("JSON yedek dosyası seç", type=["json"], key="restore_upload")
    if uploaded:
        try:
            raw = json.loads(uploaded.read().decode("utf-8"))
            # Basic validation
            if not isinstance(raw, dict):
                st.error("Geçersiz yedek formatı.")
            else:
                preview_closed = len(raw.get("closed_trades", []))
                preview_active = len(raw.get("active_positions", []))
                st.info(
                    f"Dosya içeriği: **{preview_active}** aktif pozisyon, "
                    f"**{preview_closed}** kapalı işlem"
                )
                rc1, rc2 = st.columns(2)
                with rc1:
                    if st.button("✅ Geri Yükle (mevcut veriyi yaz)", type="primary",
                                 key="confirm_restore", use_container_width=True):
                        # Ensure required keys
                        raw.setdefault("next_id", max(
                            [t.get("id", 0) for t in raw.get("active_positions", []) +
                             raw.get("closed_trades", [])], default=0
                        ) + 1)
                        save_data(raw)
                        st.session_state.data = raw
                        st.success("✅ Veri geri yüklendi.")
                        st.rerun()
                with rc2:
                    if st.button("Merge (ekle, silme)", key="merge_restore", use_container_width=True):
                        _merge_restore(data, raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            st.error(f"Dosya okunamadı: {e}")

    # ── Danger zone ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">⛔ Tehlikeli Bölge</div>', unsafe_allow_html=True)
    with st.expander("Tüm kapalı işlemleri sil (geri alınamaz)"):
        st.warning("Bu işlem kapatılmış tüm trade kayıtlarını kalıcı olarak siler!")
        confirm = st.text_input("Onaylamak için 'SİL' yazın", key="danger_confirm")
        if st.button("🗑️ Kapalı İşlemleri Temizle", key="danger_clear"):
            if confirm == "SİL":
                data["closed_trades"] = []
                save_data(data)
                st.session_state.data = data
                st.success("Kapalı işlemler silindi.")
                st.rerun()
            else:
                st.error("Onay metni hatalı.")


def _merge_restore(current: dict, incoming: dict):
    """Add incoming trades/positions without overwriting existing ones."""
    existing_ids = {p["id"] for p in current.get("active_positions", []) +
                    current.get("closed_trades", [])}
    max_id = current.get("next_id", 1)

    for t in incoming.get("closed_trades", []):
        if t.get("id") not in existing_ids:
            current["closed_trades"].append(t)
            max_id = max(max_id, t.get("id", 0) + 1)

    for t in incoming.get("active_positions", []):
        if t.get("id") not in existing_ids:
            current["active_positions"].append(t)
            max_id = max(max_id, t.get("id", 0) + 1)

    current["next_id"] = max_id
    save_data(current)
    st.session_state.data = current
    st.success(f"✅ Merge tamamlandı.")
    st.rerun()

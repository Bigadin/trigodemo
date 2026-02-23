import { useState } from 'react'
import { useEditorStore } from '@/stores/editorStore'
import styles from './ZoneEditor.module.css'

export default function EditorSidebar() {
  const video = useEditorStore((s) => s.video)
  const w = useEditorStore((s) => s.w)
  const h = useEditorStore((s) => s.h)
  const zone = useEditorStore((s) => s.zone)
  const zones = useEditorStore((s) => s.zones)
  const presence = useEditorStore((s) => s.presence)
  const polygonIdx = useEditorStore((s) => s.polygonIdx)
  const selectZone = useEditorStore((s) => s.selectZone)
  const selectPolygon = useEditorStore((s) => s.selectPolygon)
  const setTool = useEditorStore((s) => s.setTool)
  const addZone = useEditorStore((s) => s.addZone)
  const deleteZone = useEditorStore((s) => s.deleteZone)
  const deleteShape = useEditorStore((s) => s.deleteShape)
  const saveAll = useEditorStore((s) => s.saveAll)

  const [newName, setNewName] = useState('')

  const zoneKeys = Object.keys(zones).sort()
  const selectedPolys = zone ? (zones[zone]?.polygons || []) : []

  const handleAddZone = () => {
    if (!newName.trim()) return
    addZone(newName.trim())
    setNewName('')
  }

  return (
    <div className={styles.right}>
      {/* Camera info */}
      <div className={styles.card}>
        <h3>Caméra</h3>
        <div className={styles.kv}>
          <span className={styles.kvKey}>Source</span>
          <span>{video ?? '—'}</span>
          <span className={styles.kvKey}>Résol.</span>
          <span>{w}×{h}</span>
          <span className={styles.kvKey}>Status</span>
          <span>Online</span>
        </div>
      </div>

      {/* Zones list */}
      <div className={styles.card}>
        <h3>Zones</h3>
        <div className={styles.zonesList}>
          {zoneKeys.length === 0 ? (
            <div className={styles.zonesEmpty}>Aucune zone</div>
          ) : (
            zoneKeys.map((z) => {
              const info = presence[z]
              const count = (zones[z]?.polygons || []).length
              const active = zone === z
              return (
                <button
                  key={z}
                  className={`${styles.zoneRow} ${active ? styles.zoneRowActive : ''}`}
                  onClick={() => selectZone(z)}
                >
                  <div className={styles.zoneLeft}>
                    <span className={styles.zoneDot} />
                    <span className={styles.zoneName}>{z}</span>
                  </div>
                  <span className={styles.zoneMeta}>
                    {info?.formatted_time ?? '—'} · {count}
                  </span>
                </button>
              )
            })
          )}
        </div>

        {/* Add zone */}
        <div className={styles.addZoneRow}>
          <input
            className={styles.addZoneInput}
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="Nouvelle zone…"
            onKeyDown={(e) => { if (e.key === 'Enter') handleAddZone() }}
          />
          <button className={styles.addZoneBtn} onClick={handleAddZone}>+</button>
        </div>

        {zone && (
          <button className={`${styles.toolBtn} ${styles.toolBtnRed} ${styles.deleteZoneBtn}`} onClick={deleteZone}>
            Supprimer "{zone}"
          </button>
        )}
      </div>

      {/* Shape labels */}
      {selectedPolys.length > 0 && (
        <div className={styles.card}>
          <h3>Formes ({selectedPolys.length})</h3>
          <div className={styles.shapeList}>
            {selectedPolys.map((_, idx) => (
              <button
                key={idx}
                className={`${styles.shapeLabel} ${polygonIdx === idx ? styles.shapeLabelActive : ''}`}
                onClick={() => { setTool('select'); selectPolygon(idx) }}
              >
                Forme {idx + 1}
              </button>
            ))}
          </div>
          {typeof polygonIdx === 'number' && (
            <button
              className={`${styles.toolBtn} ${styles.toolBtnRed}`}
              onClick={() => deleteShape(polygonIdx)}
              style={{ marginTop: 8, width: '100%' }}
            >
              Supprimer cette forme
            </button>
          )}
        </div>
      )}

      {/* Footer */}
      <div className={styles.editorFooter}>
        <button className={`${styles.toolBtn} ${styles.toolBtnSave}`} onClick={saveAll} style={{ width: '100%' }}>
          💾 Sauvegarder tout
        </button>
      </div>
    </div>
  )
}

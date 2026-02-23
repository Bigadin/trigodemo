import { useEditorStore } from '@/stores/editorStore'
import styles from './ZoneEditor.module.css'

export default function EditorToolbar() {
  const tool = useEditorStore((s) => s.tool)
  const zone = useEditorStore((s) => s.zone)
  const points = useEditorStore((s) => s.points)
  const dirty = useEditorStore((s) => s.dirty)
  const setTool = useEditorStore((s) => s.setTool)
  const popUndo = useEditorStore((s) => s.popUndo)
  const clearTemp = useEditorStore((s) => s.clearTemp)
  const commitDraft = useEditorStore((s) => s.commitDraft)

  const isDisabled = !zone
  const hasDraft = (tool === 'include' || tool === 'exclude' || tool === 'countingROI') && points.length >= 3
  const hasModification = tool === 'select' && dirty
  const canSave = hasDraft || hasModification

  return (
    <div className={`${styles.tools} ${isDisabled ? styles.toolsDisabled : ''}`}>
      <div className={styles.toolRow}>
        <div className={styles.toolRowLeft}>
          <button
            className={`${styles.toolBtn} ${tool === 'select' ? styles.toolBtnActive : ''}`}
            onClick={() => setTool('select')}
            disabled={isDisabled}
            title="Sélection / Édition"
          >
            ◇ Sélection
          </button>
          <button
            className={`${styles.toolBtn} ${styles.toolBtnGreen} ${tool === 'include' ? styles.toolBtnActive : ''}`}
            onClick={() => { setTool('include'); }}
            disabled={isDisabled}
            title="Dessiner une zone d'inclusion"
          >
            + Include
          </button>
          <button
            className={`${styles.toolBtn} ${styles.toolBtnRed} ${tool === 'exclude' ? styles.toolBtnActive : ''}`}
            onClick={() => { setTool('exclude'); }}
            disabled={isDisabled}
            title="Dessiner une zone d'exclusion"
          >
            − Exclure
          </button>
          <button
            className={`${styles.toolBtn} ${styles.toolBtnPrimary} ${tool === 'countingROI' ? styles.toolBtnActive : ''}`}
            onClick={() => { setTool('countingROI'); }}
            disabled={isDisabled}
            title="ROI Comptage"
          >
            ▭ Comptage
          </button>

          <span className={styles.toolSep} />

          <button
            className={`${styles.toolBtn} ${styles.toolBtnCompact}`}
            onClick={popUndo}
            disabled={isDisabled}
            title="Annuler (Ctrl+Z)"
          >
            ↶
          </button>
          <button
            className={`${styles.toolBtn} ${styles.toolBtnCompact}`}
            onClick={clearTemp}
            disabled={isDisabled}
            title="Effacer le tracé"
          >
            ✕
          </button>
        </div>

        <button
          className={`${styles.toolBtn} ${styles.toolBtnSave}`}
          onClick={commitDraft}
          disabled={isDisabled || !canSave}
        >
          ✓ Valider
        </button>
      </div>

      <div className={styles.guide}>
        {tool === 'select' && 'Sélection : cliquez un point puis glissez pour déplacer. ⇧ + clic près d\'une arête = ajouter un point.'}
        {tool === 'include' && 'Zone d\'inclusion : cliquez pour placer des points (3+), puis Valider.'}
        {tool === 'exclude' && 'Zone d\'exclusion : cliquez pour placer des points (3+), puis Valider.'}
        {tool === 'countingROI' && 'ROI Comptage : dessinez un polygone (3+ points) délimitant la zone. Puis Valider.'}
      </div>
    </div>
  )
}

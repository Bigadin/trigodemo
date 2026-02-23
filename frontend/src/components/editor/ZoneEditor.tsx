import { useEditorStore } from '@/stores/editorStore'
import EditorCanvas from './EditorCanvas'
import EditorToolbar from './EditorToolbar'
import EditorSidebar from './EditorSidebar'
import styles from './ZoneEditor.module.css'

export default function ZoneEditor() {
  const open = useEditorStore((s) => s.open)
  const video = useEditorStore((s) => s.video)
  const closeEditor = useEditorStore((s) => s.closeEditor)

  if (!open) return null

  return (
    <div className={styles.overlay} onClick={(e) => { if (e.target === e.currentTarget) closeEditor() }}>
      <div className={styles.editor}>
        {/* Top bar */}
        <div className={styles.topbar}>
          <div>
            <div className={styles.title}>Video Editing — {video}</div>
            <div className={styles.subtitle}>Source : {video}</div>
          </div>
          <button className={styles.closeBtn} onClick={closeEditor} title="Fermer">
            ✕
          </button>
        </div>

        {/* Body: canvas + sidebar */}
        <div className={styles.body}>
          <div className={styles.left}>
            <EditorCanvas />
            <EditorToolbar />
          </div>
          <EditorSidebar />
        </div>
      </div>
    </div>
  )
}

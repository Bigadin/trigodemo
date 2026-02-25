import { useState, useRef, useEffect } from 'react'
import { createPortal } from 'react-dom'
import styles from './CardMenu.module.css'

export interface CardMenuOption {
  label: string
  onClick: () => void
  /** Style destructif (rouge) pour actions comme Supprimer */
  danger?: boolean
}

interface CardMenuProps {
  options: CardMenuOption[]
  className?: string
  title?: string
  /** z-index du panel (défaut 200). Utiliser 10001+ si dans un modal. */
  panelZIndex?: number
}

export default function CardMenu({ options, className = '', title = 'Options', panelZIndex = 200 }: CardMenuProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [panelPos, setPanelPos] = useState({ top: 0, left: 0 })
  const rootRef = useRef<HTMLDivElement>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const panelRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      const target = e.target as Node
      if (
        rootRef.current?.contains(target) ||
        panelRef.current?.contains(target)
      ) {
        return
      }
      setIsOpen(false)
    }
    document.addEventListener('click', onDocClick, true)
    return () => document.removeEventListener('click', onDocClick, true)
  }, [])

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsOpen(false)
    }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [])

  useEffect(() => {
    if (!isOpen || !triggerRef.current) return
    const rect = triggerRef.current.getBoundingClientRect()
    setPanelPos({
      top: rect.bottom + 4,
      left: rect.right - 140,
    })
  }, [isOpen])

  return (
    <div ref={rootRef} className={`${styles.wrap} ${className}`}>
      <button
        ref={triggerRef}
        type="button"
        className={styles.trigger}
        onClick={(e) => {
          e.preventDefault()
          e.stopPropagation()
          setIsOpen((o) => !o)
        }}
        title={title}
      >
        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
          <circle cx="8" cy="2.5" r="1.5" />
          <circle cx="8" cy="8" r="1.5" />
          <circle cx="8" cy="13.5" r="1.5" />
        </svg>
      </button>
      {isOpen &&
        createPortal(
          <div
            ref={(el) => {
              panelRef.current = el
            }}
            className={styles.panel}
            style={{
              position: 'fixed',
              top: panelPos.top,
              left: panelPos.left,
              minWidth: 140,
              zIndex: panelZIndex,
            }}
          >
            {options.map((opt) => (
              <button
                key={opt.label}
                type="button"
                className={`${styles.option} ${opt.danger ? styles.optionDanger : ''}`}
                onClick={(e) => {
                  e.stopPropagation()
                  opt.onClick()
                  setIsOpen(false)
                }}
              >
                {opt.label}
              </button>
            ))}
          </div>,
          document.body
        )}
    </div>
  )
}

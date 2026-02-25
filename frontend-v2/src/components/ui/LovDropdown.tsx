import { useState, useRef, useEffect } from 'react'
import styles from './LovDropdown.module.css'

export interface LovOption {
  value: string
  label: string
}

interface LovDropdownProps {
  options: LovOption[]
  value: string
  onChange: (value: string) => void
  placeholder?: string
  emptyLabel?: string
  disabled?: boolean
  className?: string
  fullWidth?: boolean
  /** Style inline : pas de contour ni flèche par défaut, hover discret, ouverture au clic */
  inline?: boolean
  /** Style formulaire : harmonie avec les champs input (même bordure, fond, padding) */
  form?: boolean
  /** Icône à afficher dans le trigger (ex. pour inline : icône caméra) */
  icon?: React.ReactNode
}

export default function LovDropdown({
  options,
  value,
  onChange,
  placeholder = 'Sélectionner…',
  emptyLabel = 'Aucune option',
  disabled = false,
  className = '',
  fullWidth = false,
  inline = false,
  form = false,
  icon: iconProp,
}: LovDropdownProps) {
  const [isOpen, setIsOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)

  const selected = options.find((o) => o.value === value)
  const label = selected?.label ?? placeholder
  const isPlaceholder = !selected

  useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setIsOpen(false)
      }
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

  return (
    <div
      ref={rootRef}
      className={`${styles.lovDropdown} ${fullWidth ? styles.fullWidth : ''} ${inline ? styles.inline : ''} ${form ? styles.form : ''} ${isOpen ? styles.isOpen : ''} ${className}`}
    >
      <button
        type="button"
        className={styles.lovTrigger}
        onClick={() => !disabled && setIsOpen((o) => !o)}
        disabled={disabled}
      >
        {iconProp && <span className={styles.lovTriggerIcon}>{iconProp}</span>}
        <span className={`${styles.lovTriggerText} ${isPlaceholder ? styles.placeholder : ''}`}>
          {options.length === 0 ? emptyLabel : label}
        </span>
        {!inline && (
          <svg
            className={styles.lovChevron}
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        )}
      </button>
      <div className={styles.lovPanel}>
        {options.length === 0 ? (
          <div className={styles.lovOptionEmpty}>{emptyLabel}</div>
        ) : (
          options.map((opt, i) => (
            <button
              key={opt.value}
              type="button"
              className={`${styles.lovOption} ${opt.value === value ? styles.isSelected : ''}`}
              style={{ ['--i' as string]: i }}
              onClick={() => {
                onChange(opt.value)
                setIsOpen(false)
              }}
            >
              {opt.label}
            </button>
          ))
        )}
      </div>
    </div>
  )
}

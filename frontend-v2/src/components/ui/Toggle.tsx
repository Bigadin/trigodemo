import React from 'react'
import { cn } from '@/lib/utils'
import styles from './Toggle.module.css'

const variantStyles: Record<string, string> = {
  default: styles.variantDefault,
  success: styles.variantSuccess,
  warning: styles.variantWarning,
  danger: styles.variantDanger,
  primary: styles.variantPrimary,
}

export interface ToggleProps {
  checked?: boolean
  onCheckedChange?: (checked: boolean) => void
  className?: string
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'primary'
  title?: string
}

export function Toggle({
  checked = false,
  onCheckedChange,
  className,
  variant = 'default',
  title,
}: ToggleProps) {
  const [isChecked, setIsChecked] = React.useState(checked)
  const filterId = React.useId().replace(/:/g, '')

  React.useEffect(() => {
    setIsChecked(checked)
  }, [checked])

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setIsChecked(e.target.checked)
    onCheckedChange?.(e.target.checked)
  }

  return (
    <label className={cn(styles.switch, variantStyles[variant], className)} title={title}>
      <svg className={styles.svgFilter} aria-hidden>
        <defs>
          <filter id={filterId} x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur in="SourceGraphic" stdDeviation="2" result="blur" />
            <feColorMatrix
              in="blur"
              mode="matrix"
              values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 18 -7"
              result="goo"
            />
            <feComposite in="SourceGraphic" in2="goo" operator="atop" />
          </filter>
        </defs>
      </svg>
      <input
        type="checkbox"
        checked={isChecked}
        onChange={handleChange}
        className={styles.input}
      />
      <svg viewBox="0 0 52 32" filter={`url(#${filterId})`} className={styles.svg}>
        <circle
          className={styles.circle}
          cx="16"
          cy="16"
          r="10"
          style={{
            transform: isChecked ? 'translateX(12px) scale(0)' : 'translateX(0) scale(1)',
            transformOrigin: '16px 16px',
          }}
        />
        <circle
          className={styles.circle}
          cx="36"
          cy="16"
          r="10"
          style={{
            transform: isChecked ? 'translateX(0) scale(1)' : 'translateX(-12px) scale(0)',
            transformOrigin: '36px 16px',
          }}
        />
        {isChecked && (
          <circle className={styles.dropCircle} cx="35" cy="-1" r="2.5" />
        )}
      </svg>
    </label>
  )
}

export function GooeyFilter() {
  return (
    <svg aria-hidden style={{ position: 'absolute', width: 0, height: 0, overflow: 'visible' }}>
      <defs>
        <filter id="goo" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="2" result="blur" />
          <feColorMatrix
            in="blur"
            mode="matrix"
            values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 18 -7"
            result="goo"
          />
          <feComposite in="SourceGraphic" in2="goo" operator="atop" />
        </filter>
      </defs>
    </svg>
  )
}

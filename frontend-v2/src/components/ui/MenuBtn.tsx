export default function MenuBtn({
  className,
  onClick,
  title = 'Options',
}: {
  className?: string
  onClick?: (e: React.MouseEvent) => void
  title?: string
}) {
  return (
    <button className={className} type="button" onClick={onClick} title={title}>
      <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
        <circle cx="8" cy="2.5" r="1.5" />
        <circle cx="8" cy="8" r="1.5" />
        <circle cx="8" cy="13.5" r="1.5" />
      </svg>
    </button>
  )
}

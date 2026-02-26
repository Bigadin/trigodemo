import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { HierarchyProvider } from '@/context/HierarchyContext'
import { SessionProvider } from '@/context/SessionContext'
import { ErrorBoundary } from '@/components/ErrorBoundary'
import App from './App'
import './styles/tokens.css'
import './styles/reset.css'

const root = document.getElementById('root')
if (!root) {
  document.body.innerHTML = '<div style="padding:24px;font-family:sans-serif">Erreur: #root introuvable</div>'
} else {
  ReactDOM.createRoot(root).render(
    <React.StrictMode>
      <ErrorBoundary>
        <BrowserRouter>
          <HierarchyProvider>
            <SessionProvider>
              <App />
            </SessionProvider>
          </HierarchyProvider>
        </BrowserRouter>
      </ErrorBoundary>
    </React.StrictMode>,
  )
}

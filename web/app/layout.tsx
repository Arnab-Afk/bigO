import React from "react"
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { Analytics } from '@vercel/analytics/next'
import './globals.css'

const inter = Inter({ subsets: ["latin"], variable: '--font-inter' });

export const metadata: Metadata = {
  title: 'RUDRA - Resilient Unified Decision & Risk Analytics',
  description: 'Network-based game-theoretic modeling of financial infrastructure. Predict systemic risk, prevent cascading failures, and ensure financial stability.',
  generator: 'Arnab Bhowmik',
  icons: {
    icon: [
      {
        url: '/images/hero-mono-overlay.png',
        media: '(prefers-color-scheme: light)',
      },
      {
        url: '/images/hero-mono-overlay.png',
        media: '(prefers-color-scheme: dark)',
      },
      {
        url: '/images/hero-mono-overlay.png',
        type: 'image/svg+xml',
      },
    ],
    apple: '/images/hero-mono-overlay.png',
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} font-sans antialiased`}>
        {children}
        <Analytics />
      </body>
    </html>
  )
}

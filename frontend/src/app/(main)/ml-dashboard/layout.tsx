"use client";

import { ReactNode } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import {
  LayoutDashboard,
  Network,
  Building2,
  Play,
  AlertTriangle,
  FileText,
  BarChart3,
} from 'lucide-react';

const navigation = [
  { name: 'Overview', href: '/ml-dashboard', icon: LayoutDashboard },
  { name: 'Network', href: '/ml-dashboard/network', icon: Network },
  { name: 'Banks', href: '/ml-dashboard/banks', icon: Building2 },
  { name: 'Simulation', href: '/ml-dashboard/simulation', icon: Play },
  { name: 'Stress Test', href: '/ml-dashboard/stress-test', icon: AlertTriangle },
  { name: 'Reports', href: '/ml-dashboard/reports', icon: FileText },
];

export default function MLDashboardLayout({
  children,
}: {
  children: ReactNode;
}) {
  const pathname = usePathname();

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <aside className="w-64 border-r bg-card">
        <div className="flex h-full flex-col">
          {/* Logo/Header */}
          <div className="flex h-16 items-center border-b px-6">
            <BarChart3 className="h-6 w-6 text-primary" />
            <span className="ml-2 text-lg font-semibold">RUDRA ML</span>
          </div>

          {/* Navigation */}
          <nav className="flex-1 space-y-1 px-3 py-4">
            {navigation.map((item) => {
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-primary text-primary-foreground'
                      : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                  )}
                >
                  <item.icon className="h-5 w-5" />
                  {item.name}
                </Link>
              );
            })}
          </nav>

          {/* Footer */}
          <div className="border-t p-4">
            <p className="text-xs text-muted-foreground">
              CCP Risk Analysis Platform
            </p>
            <p className="text-xs text-muted-foreground">Version 1.0.0</p>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">
        <div className="container mx-auto p-6">{children}</div>
      </main>
    </div>
  );
}

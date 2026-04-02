import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Providers } from "@/components/providers";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "DR Detection - Diabetic Retinopathy Screening",
  description: "AI-powered diabetic retinopathy detection and grading from retinal fundus images",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-background text-foreground">
        <Providers>
          <nav className="border-b border-gray-200 dark:border-gray-800 px-6 py-4">
            <div className="max-w-6xl mx-auto flex items-center justify-between">
              <a href="/" className="text-xl font-bold">
                DR Detection
              </a>
              <div className="flex gap-6 text-sm">
                <a href="/" className="hover:text-blue-600 transition-colors">
                  Screening
                </a>
                <a href="/dashboard" className="hover:text-blue-600 transition-colors">
                  Dashboard
                </a>
              </div>
            </div>
          </nav>
          <main className="flex-1">{children}</main>
          <footer className="border-t border-gray-200 dark:border-gray-800 px-6 py-4 text-center text-xs text-gray-400">
            Automated Diabetic Retinopathy Detection System
          </footer>
        </Providers>
      </body>
    </html>
  );
}

import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import BootSplash from "@/components/BootSplash";
import "./globals.css";

export const metadata: Metadata = {
  title: "Centaur",
  description: "Financial Document Intelligence Platform",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning className={`${GeistSans.variable} ${GeistMono.variable}`}>
      <body>
        <BootSplash />
        {children}
      </body>
    </html>
  );
}

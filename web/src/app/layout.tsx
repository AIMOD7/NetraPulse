import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "NetraPulse — DR Screening",
  description:
    "AI-powered diabetic retinopathy screening. Upload a fundus image to get an instant DR grade and Grad-CAM explanation.",
  keywords: ["diabetic retinopathy", "fundus", "AI screening", "Grad-CAM"],
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>{children}</body>
    </html>
  );
}

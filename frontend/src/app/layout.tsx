import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "NemoReconstruct",
  description: "Video to 3D Gaussian reconstruction (NuRec USDZ + PLY) for Omniverse and Isaac Sim workflows.",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

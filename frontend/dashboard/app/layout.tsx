import "./globals.css";
import Link from "next/link";

export const metadata = {
  title: "Robot Navigation RL + KAN",
  description: "Simulation platform dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <nav className="nav">
          <span className="brand">RobotNav KAN</span>
          <Link href="/">Overview</Link>
          <Link href="/training">Training</Link>
          <Link href="/live">Live Simulation</Link>
          <Link href="/explain">Explainability</Link>
        </nav>
        <main className="container">{children}</main>
      </body>
    </html>
  );
}
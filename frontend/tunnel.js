import ngrok from '@ngrok/ngrok';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Read .env from parent directory
const envPath = path.resolve(__dirname, '../.env');
let authtoken = '';
if (fs.existsSync(envPath)) {
  const envContent = fs.readFileSync(envPath, 'utf-8');
  for (const line of envContent.split('\n')) {
    const match = line.match(/^\s*NGROK_AUTHTOKEN\s*=\s*["']?([^"'\r\n]+)["']?/);
    if (match) {
      authtoken = match[1].trim();
      break;
    }
  }
}

if (!authtoken) {
  console.error('Error: NGROK_AUTHTOKEN not found in .env');
  process.exit(1);
}

const domain = 'regular-chief-hastiness.ngrok-free.dev';

async function startTunnel() {
  try {
    console.log(`Connecting to ngrok with domain ${domain}...`);
    const listener = await ngrok.forward({
      addr: 8000,
      authtoken: authtoken,
      domain: domain,
    });
    console.log(`SUCCESS: Permanent ngrok tunnel active at: ${listener.url()}`);
    // Keep process alive
    process.stdin.resume();
  } catch (err) {
    console.error('Failed to establish ngrok tunnel:', err);
    process.exit(1);
  }
}

startTunnel();


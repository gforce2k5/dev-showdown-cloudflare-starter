import { SELF } from 'cloudflare:test';
import { describe, expect, it } from 'vitest';

describe('Worker', () => {
	it('shows a helpful health-check message at the root URL', async () => {
		const response = await SELF.fetch('https://example.com');

		expect(response.status).toBe(200);
		expect(response.headers.get('Content-Type')).toContain('text/plain');

		const text = await response.text();
		expect(text).toBe('Dev Showdown Cloudflare Starter is running.');
	});

	it('solves HELLO_WORLD through the worker runtime', async () => {
		const response = await SELF.fetch('https://example.com/api?challengeType=HELLO_WORLD', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'X-Interaction-Id': 'integration-test',
			},
			body: JSON.stringify({ name: 'Bob' }),
		});

		expect(response.status).toBe(200);
		expect(await response.json()).toEqual({
			greeting: 'Hello Bob',
		});
	});
});

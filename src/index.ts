import { createOpenAICompatible, } from '@ai-sdk/openai-compatible';
import { generateObject, generateText, streamText, tool, stepCountIs } from 'ai';
import { z } from 'zod';

const INTERACTION_ID_HEADER = 'X-Interaction-Id';

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		const url = new URL(request.url);

		if (request.method === 'GET' && url.pathname === '/') {
			const runningMessage = 'Dev Showdown Cloudflare Starter is running.';
			const message = env.DEV_SHOWDOWN_API_KEY
				? runningMessage
				: [runningMessage, 'DEV_SHOWDOWN_API_KEY is missing.'].join(
						'\n',
					);

			return new Response(message, {
				headers: {
					'Content-Type': 'text/plain; charset=utf-8',
				},
			});
		}

		if (request.method !== 'POST' || url.pathname !== '/api') {
			return new Response('Not Found', { status: 404 });
		}

		const challengeType = url.searchParams.get('challengeType');
		if (!challengeType) {
			return new Response('Missing challengeType query parameter', {
				status: 400,
			});
		}

		const interactionId = request.headers.get(INTERACTION_ID_HEADER);
		if (!interactionId) {
			return new Response(`Missing ${INTERACTION_ID_HEADER} header`, {
				status: 400,
			});
		}

		const payload = await request.json<any>();

		switch (challengeType) {
			case 'HELLO_WORLD':
				return Response.json({
					greeting: `Hello ${payload.name}`,
				});
			case 'BASIC_LLM': {
				if (!env.DEV_SHOWDOWN_API_KEY) {
					throw new Error('DEV_SHOWDOWN_API_KEY is required');
				}

				const workshopLlm = createWorkshopLlm(env.DEV_SHOWDOWN_API_KEY, interactionId);
				const result = await generateText({
					model: workshopLlm.chatModel('deli-4'),
					system: 'You are a trivia question player. Answer the question correctly and concisely.',
					prompt: payload.question,
				});

				return Response.json({
					answer: result.text || 'N/A',
				});
			}
			case 'JSON_MODE': {
				if (!env.DEV_SHOWDOWN_API_KEY) {
					throw new Error('DEV_SHOWDOWN_API_KEY is required');
				}

				const workshopLlm = createWorkshopLlm(env.DEV_SHOWDOWN_API_KEY, interactionId);
				const result = await generateObject({
					model: workshopLlm.chatModel('deli-4'),
					schemaName: 'product',
					schema: z.object({
						name: z.string(),
						price: z.number(),
						currency: z.string(),
						inStock: z.boolean(),
						dimensions: z.object({
							length: z.number(),
							width: z.number(),
							height: z.number(),
							unit: z.string(),
						}),
						manufacturer: z.object({
							name: z.string(),
							country: z.string(),
							website: z.string(),
						}),
						specifications: z.object({
							weight: z.number(),
							weightUnit: z.string(),
							warrantyMonths: z.number(),
						}),
					}),
					system:
						'You will be given a human-readable description of a product and must return the extracted data as a JSON object. The wording and sentence order will vary, but every required fact is present in the text.',
					prompt: payload.description,
				});

				return Response.json(result.object);
			}
			case 'BASIC_TOOL_CALL': {
				if (!env.DEV_SHOWDOWN_API_KEY) {
					throw new Error('DEV_SHOWDOWN_API_KEY is required');
				}

				const workshopLlm = createWorkshopLlm(env.DEV_SHOWDOWN_API_KEY, interactionId);
				const result = await generateText({
					model: workshopLlm.chatModel('deli-4'),
					system:
						'You are a helpful weather assistant. When the user asks about the weather in a city, use the get_weather tool to fetch the current weather, then respond in natural language and include the temperature returned by the tool.',
					prompt: payload.question,
					tools: {
						get_weather: tool({
							description: 'Get the current weather for a given city.',
							inputSchema: z.object({
								city: z.string().describe('The name of the city to get the weather for.'),
							}),
							execute: async ({ city }) => {
								const response = await fetch('https://devshowdown.com/api/weather', {
									method: 'POST',
									headers: {
										'Content-Type': 'application/json',
										[INTERACTION_ID_HEADER]: interactionId,
									},
									body: JSON.stringify({ city }),
								});
								if (!response.ok) {
									throw new Error(`Weather API error: ${response.status}`);
								}
								return await response.json();
							},
						}),
					},
					stopWhen: stepCountIs(5),
				});

				return Response.json({
					answer: result.text || 'N/A',
				});
			}
			case 'RESPONSE_STREAMING': {
				if (!env.DEV_SHOWDOWN_API_KEY) {
					throw new Error('DEV_SHOWDOWN_API_KEY is required');
				}

				const workshopLlm = createWorkshopLlm(env.DEV_SHOWDOWN_API_KEY, interactionId);
				const result = streamText({
					model: workshopLlm.chatModel('deli-4'),
					prompt: payload.prompt,
				});

				const encoder = new TextEncoder();
				const textStream = result.textStream;
				const body = new ReadableStream<Uint8Array>({
					async start(controller) {
						try {
							controller.enqueue(encoder.encode('"'));
							for await (const chunk of textStream) {
								if (!chunk) continue;
								const escaped = JSON.stringify(chunk).slice(1, -1);
								controller.enqueue(encoder.encode(escaped));
							}
							controller.enqueue(encoder.encode('"'));
							controller.close();
						} catch (err) {
							controller.error(err);
						}
					},
				});

				return new Response(body, {
					headers: {
						'Content-Type': 'application/json; charset=utf-8',
						'Transfer-Encoding': 'chunked',
						'Cache-Control': 'no-cache, no-transform',
					},
				});
			}
			default:
				return new Response('Solver not found', { status: 404 });
			}
	},
	} satisfies ExportedHandler<Env>;

function createWorkshopLlm(apiKey: string, interactionId: string) {
	return createOpenAICompatible({
		name: 'dev-showdown',
		baseURL: 'https://devshowdown.com/v1',
		supportsStructuredOutputs: true,
		headers: {
			Authorization: `Bearer ${apiKey}`,
			[INTERACTION_ID_HEADER]: interactionId,
		},
	});
}

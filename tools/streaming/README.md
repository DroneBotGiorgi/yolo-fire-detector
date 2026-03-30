# Streaming Tools

Componenti locali per simulare uno stream video e testare il detector contro RTMP/HLS.

## Avvio publisher webcam 2

```bash
python tools/streaming/fake_rtmp_webcam.py --camera 2
```

## Endpoint disponibili

- RTMP: `rtmp://127.0.0.1:1935/webcam2`
- HLS: `http://127.0.0.1:8888/webcam2/index.m3u8`

## Come visualizzare lo stream

Nota: per preview locale quasi real-time conviene RTMP. HLS e' piu' lento per natura perche' segmenta e bufferizza il video.

Con FFplay:

```bash
ffplay rtmp://127.0.0.1:1935/webcam2
```

Oppure con HLS:

```bash
ffplay http://127.0.0.1:8888/webcam2/index.m3u8
```

Se vuoi ridurre ancora il lag di preview locale, usa RTMP con FFplay e non HLS.

## Come testare il detector

```bash
python detect.py --source rtmp://127.0.0.1:1935/webcam2
```
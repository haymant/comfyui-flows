# Prompt examples

Copy these into the prompt box. H3 wants one block: scene, timeline, camera,
audio. SDXL wants a comma-separated visual description.

## t2v — paper boat

```text
A five-second cinematic shot at golden hour: a small white paper boat drifts
across a rain puddle on a quiet city street. Warm low sun rakes across wet
asphalt, reflections ripple, a light breeze moves fallen leaves. The camera
starts low and close to the water, then slowly tracks backward and rises to
reveal the street and a distant row of buildings.

One continuous take, no cuts. Shallow depth of field, natural film grain, soft
anamorphic highlights.

Audio: gentle rain dripping, distant traffic, a single warm piano note, and a
quiet low strings bed. No speech, no music stings.
```

## i2v — animate a photo
```text
Animate the provided first frame as a five-second continuous shot. Keep the
subject, face, clothing colours, composition and lighting exactly consistent
with the input image. Add only subtle natural motion: a slow breath, a blink,
fabric moving in a light breeze, and fine environmental movement in the
background.

Camera: slow, smooth push-in. No cuts, no morphing, no new objects.

Audio: quiet room tone and soft natural ambience that matches the scene. No speech.
```

## flf2v — start and end frame

```text
Create a five-second continuous shot that starts exactly on the first frame and
ends exactly on the last frame. Keep the subject, face, clothing colours,
lighting and overall style consistent between both images. Generate only the
natural motion and camera movement needed to travel smoothly from the start
composition to the end composition.

One continuous take. No cuts, no morphing, no new objects. Camera: smooth,
steady movement. Audio: quiet ambience that matches the scene. No speech.
```

## r2v — keep a face

```text
Use <Picture 1> as the identity reference for the main subject. Keep the same
face, hair, body proportions and outfit. Place the subject in a new five-second
cinematic shot: they turn toward the camera in a softly lit room, hold eye
contact, and give a small, natural smile.

One continuous take. The camera slowly circles a few degrees; shallow depth of
field; natural skin texture; soft window light.

Audio: quiet room tone, a faint fabric rustle, and one soft piano note. No speech.
```

## t2i — SDXL reading corner

```text
A calm editorial photograph of a sunlit reading corner, a linen armchair beside
a tall window, a small stack of books and a ceramic cup on a wooden side table,
soft morning light, muted warm palette, fine film grain, shallow depth of field,
natural textures, no text, no people.
```

## Negative prompt for SDXL

```text
text, watermark, logo, signature, low quality, blurry, oversaturated, deformed,
extra limbs, cartoon, cgi, plastic skin, harsh flash, cluttered composition
```

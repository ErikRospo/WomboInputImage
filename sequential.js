const task = require("./index.js");
const Rest = require("./rest.js");
const styles = require("./styles.js");
const download = require("./download.js");
const fs = require("fs");

let settings = JSON.parse(fs.readFileSync("./settings.json"))
let quiet = settings.quiet||true;
let inter = settings.inter||false;
let final = settings.final||true;
let images = [];
let Siterations=settings.iterations
let styleSelect = settings.style
if (typeof styleSelect === "string") {
  styleSelect = Array(Siterations).fill(styleSelect)
}
let p = settings.inputImages
if (typeof p === "string") {
  p = Array(Siterations).fill(p)
} 
let prompt = settings.prompt
if (typeof prompt === "string") {
  prompt = Array(Siterations).fill(prompt)
}
let file_folder=settings.file_folder
for (let n = 0; n < p.length; n++) {
  images.push(fs.readFileSync(p[n]).toString("base64"));
}

async function generate(prompt, style, prefix, input_image = false, download_dir = "./generated", iteration_ = 0) {
  function handler(data, prefix) {
    switch (data.state) {
      case "authenticated":
        if (!quiet) console.log(`${prefix}Authenticated, allocating a task...`);
        break;
      case "allocated":
        if (!quiet)
          console.log(`${prefix}Allocated, submitting the prompt and style...`);
        break;
      case "submitted":
        if (!quiet) console.log(`${prefix}Submitted! Waiting on results...`);
        break;
      case "progress":
        let current = data.task.photo_url_list.length;
        let max = styles.steps.get(style) + 1;
        if (!quiet)
          console.log(
            `${prefix}Submitted! Waiting on results... (${current}/${max})`
          );
        break;
      case "generated":
        if (!quiet)
          console.log(
            `${prefix}Results are in, downloading the final image...`
          );
        break;
      case "downloaded":
        if (!quiet) console.log(`${prefix}Downloaded!`);
        break;
    }
  }

  let res = await task(
    prompt,
    style,
    data => handler(data, prefix),
    { final, inter, download_dir },
    input_image, iteration_
  );
  if (!quiet && final)
    console.log(
      `${prefix}Your results have been downloaded to the following files:`
    );
  else if (!quiet)
    console.log(
      `${prefix}Task finished, the results are available at the following addresses:`
    );
  if (!quiet) {
    for (let inter of res.inter) {
      console.log(inter);
    }
    if (final) console.log(res.path);
    else console.log(res.url);
  }

  return res;
}


async function generate_sequential(prompts, styles, times, directory = Date.now()) {
  let last_image = {};
  const download_dir = `./generated/${directory}/`
  for (let n = 0; n < times; n++) {
    console.log(`${n + 1}/${times} Started`)
    await generate(prompts[n], style, `${n + 1}: `, last_image, download_dir, n);
    last_image = {
      image_weight: "MEDIUM",
      media_suffix: "jpeg",
      input_image: images[n]
    };
    console.log(`${n + 1}/${times} Finished`)

  }
}
if (require.main === module) {
  generate_sequential(prompts, styles,Siterations, file_folder);
}
module.exports.generate = generate;
module.exports.generate_sequential = generate_sequential;
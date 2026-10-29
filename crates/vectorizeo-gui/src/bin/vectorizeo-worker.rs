//! The web build's Web Worker: runs the [`Task`]s the page sends (postcard bytes) off the UI
//! thread and answers with [`Reply`]s. Trunk builds it next to the app (`index.html`).

#[cfg(target_arch = "wasm32")]
fn main() {
    use js_sys::Uint8Array;
    use vectorizeo_gui::task::{Reply, Task};
    use wasm_bindgen::JsCast;
    use wasm_bindgen::prelude::Closure;
    use web_sys::{DedicatedWorkerGlobalScope, MessageEvent};

    let scope: DedicatedWorkerGlobalScope = js_sys::global().unchecked_into();
    let post = {
        let scope = scope.clone();
        move |reply: &Reply| {
            let bytes = postcard::to_allocvec(reply).expect("replies serialize");
            let data = Uint8Array::from(&bytes[..]);
            let _ = scope.post_message_with_transfer(&data, &js_sys::Array::of1(&data.buffer()));
        }
    };
    let onmessage = {
        let post = post.clone();
        Closure::<dyn FnMut(MessageEvent)>::new(move |ev: MessageEvent| {
            let bytes = Uint8Array::new(&ev.data()).to_vec();
            let task: Task = postcard::from_bytes(&bytes).expect("a task from the page");
            let out = task.run(&mut |label| post(&Reply::Stage(label.to_owned())));
            post(&Reply::Done(out));
        })
    };
    scope.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
    onmessage.forget();
    post(&Reply::Ready);
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!("vectorizeo-worker is the web build's Web Worker; run vectorizeo-gui instead");
}

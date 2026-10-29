//! Background tasks: image loading through Bézier fitting, one task per stage.
//!
//! Natively every [`Task`] runs on its own thread. Starting a task or cancelling bumps the
//! generation: the running thread is detached and finishes in the background, and its result is
//! dropped. The engine has no cancellation points.
//!
//! On the web each `Worker` owns a Web Worker running `vectorizeo-worker` (the same task code); tasks
//! and replies travel postcard-encoded. A running task is stopped for real: starting another task
//! or cancelling terminates the Web Worker, and the next task starts a fresh one.

use std::sync::mpsc::{self, Receiver, Sender};

use eframe::egui;
use vectorizeo_gui::task::{Output, Task};
use web_time::Instant;

enum Event {
    Stage(String),
    Done(Output),
}

/// The running task.
pub struct Busy {
    pub label: String,
    pub started: Instant,
}

pub struct Worker {
    ctx: egui::Context,
    tx: Sender<(u64, Event)>,
    rx: Receiver<(u64, Event)>,
    generation: u64,
    busy: Option<Busy>,
    #[cfg(target_arch = "wasm32")]
    remote: web::Remote,
}

impl Worker {
    pub fn new(ctx: &egui::Context) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            ctx: ctx.clone(),
            tx,
            rx,
            generation: 0,
            busy: None,
            #[cfg(target_arch = "wasm32")]
            remote: web::Remote::default(),
        }
    }

    pub fn busy(&self) -> Option<&Busy> {
        self.busy.as_ref()
    }

    pub fn is_busy(&self) -> bool {
        self.busy.is_some()
    }

    /// Run `task`; a running task is detached (native) or stopped (web).
    pub fn spawn(&mut self, task: Task) {
        self.generation += 1;
        let generation = self.generation;
        self.busy = Some(Busy { label: task.label().to_owned(), started: Instant::now() });
        let (tx, ctx) = (self.tx.clone(), self.ctx.clone());
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || {
            let mut stage = |label: &str| {
                let _ = tx.send((generation, Event::Stage(label.to_owned())));
                ctx.request_repaint();
            };
            let result = task.run(&mut stage);
            let _ = tx.send((generation, Event::Done(result)));
            ctx.request_repaint();
        });
        #[cfg(target_arch = "wasm32")]
        self.remote.run(task, move |event| {
            let _ = tx.send((generation, event));
            ctx.request_repaint();
        });
    }

    /// Drop the running task's result (native: detach it; web: stop it).
    pub fn cancel(&mut self) {
        self.generation += 1;
        if self.busy.take().is_some() {
            #[cfg(target_arch = "wasm32")]
            self.remote.terminate();
        }
    }

    /// Take progress and the finished result of the current task.
    pub fn poll(&mut self) -> Option<Output> {
        let mut done = None;
        while let Ok((generation, event)) = self.rx.try_recv() {
            if generation != self.generation {
                continue;
            }
            match event {
                Event::Stage(label) => {
                    if let Some(b) = &mut self.busy {
                        b.label = label;
                    }
                }
                Event::Done(m) => {
                    self.busy = None;
                    done = Some(m);
                }
            }
        }
        done
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use std::cell::RefCell;
    use std::rc::Rc;

    use js_sys::{Array, Uint8Array};
    use vectorizeo_gui::task::{Output, Reply, Task};
    use wasm_bindgen::JsCast;
    use wasm_bindgen::prelude::Closure;
    use web_sys::{ErrorEvent, MessageEvent, WorkerOptions, WorkerType};

    use super::Event;

    /// The worker script trunk generates for `vectorizeo-worker` (`data-loader-shim` in index.html).
    const SCRIPT: &str = "./vectorizeo-worker_loader.js";

    type Sink = Rc<dyn Fn(Event)>;

    struct State {
        ready: bool,
        /// A task sent before the worker was ready.
        queued: Option<Vec<u8>>,
        sink: Option<Sink>,
    }

    /// One Web Worker, created on first use and after every [`Remote::terminate`].
    #[derive(Default)]
    pub struct Remote {
        live: Option<Live>,
    }

    struct Live {
        worker: web_sys::Worker,
        state: Rc<RefCell<State>>,
        _onmessage: Closure<dyn FnMut(MessageEvent)>,
        _onerror: Closure<dyn FnMut(ErrorEvent)>,
    }

    fn post(worker: &web_sys::Worker, bytes: &[u8]) {
        let data = Uint8Array::from(bytes);
        let _ = worker.post_message_with_transfer(&data, &Array::of1(&data.buffer()));
    }

    impl Remote {
        /// Send `task`; `sink` gets its progress and result.
        pub fn run(&mut self, task: Task, sink: impl Fn(Event) + 'static) {
            // The worker runs one task at a time: a busy one is replaced.
            if self.live.as_ref().is_some_and(|l| l.state.borrow().sink.is_some()) {
                self.terminate();
            }
            let bytes = postcard::to_allocvec(&task).expect("tasks serialize");
            let live = self.live.get_or_insert_with(Live::start);
            let mut state = live.state.borrow_mut();
            state.sink = Some(Rc::new(sink));
            if state.ready {
                post(&live.worker, &bytes);
            } else {
                state.queued = Some(bytes);
            }
        }

        pub fn terminate(&mut self) {
            if let Some(live) = self.live.take() {
                live.worker.terminate();
            }
        }
    }

    impl Live {
        fn start() -> Self {
            let opts = WorkerOptions::new();
            opts.set_type(WorkerType::Module);
            let worker = web_sys::Worker::new_with_options(SCRIPT, &opts).expect("start the web worker");
            let state = Rc::new(RefCell::new(State { ready: false, queued: None, sink: None }));

            let (st, w) = (state.clone(), worker.clone());
            let onmessage = Closure::<dyn FnMut(MessageEvent)>::new(move |ev: MessageEvent| {
                let bytes = Uint8Array::new(&ev.data()).to_vec();
                let reply = match postcard::from_bytes::<Reply>(&bytes) {
                    Ok(r) => r,
                    Err(e) => Reply::Done(Output::Failed(format!("Bad reply from the web worker: {e}"))),
                };
                let mut s = st.borrow_mut();
                match reply {
                    Reply::Ready => {
                        s.ready = true;
                        if let Some(task) = s.queued.take() {
                            post(&w, &task);
                        }
                    }
                    Reply::Stage(label) => {
                        if let Some(sink) = s.sink.clone() {
                            drop(s);
                            sink(Event::Stage(label));
                        }
                    }
                    Reply::Done(out) => {
                        if let Some(sink) = s.sink.take() {
                            drop(s);
                            sink(Event::Done(out));
                        }
                    }
                }
            });
            worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));

            // A panic in the worker ends the task with an error.
            let st = state.clone();
            let onerror = Closure::<dyn FnMut(ErrorEvent)>::new(move |ev: ErrorEvent| {
                let sink = st.borrow_mut().sink.take();
                if let Some(sink) = sink {
                    sink(Event::Done(Output::Failed(format!("The web worker failed: {}", ev.message()))));
                }
            });
            worker.set_onerror(Some(onerror.as_ref().unchecked_ref()));

            Live { worker, state, _onmessage: onmessage, _onerror: onerror }
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::sync::Arc;
    use vectorizeo_core::{Rgba, RgbaImage};
    use vectorizeo_pipeline::Flatten;

    fn preview_task(v: u8) -> Task {
        let img = RgbaImage::new(1, 1, vec![Rgba { r: v, g: v, b: v, a: 255 }]);
        Task::Preview { loaded: Arc::new(img), flatten: Flatten::WHITE, palette: None }
    }

    /// Only the current task's result arrives; a cancelled one is dropped.
    #[test]
    fn cancelled_results_are_dropped() {
        let mut w = Worker::new(&egui::Context::default());
        w.spawn(preview_task(1));
        w.cancel();
        assert!(!w.is_busy());
        w.spawn(preview_task(2));
        let start = Instant::now();
        let got = loop {
            if let Some(v) = w.poll() {
                break v;
            }
            assert!(start.elapsed().as_secs() < 5, "worker result never arrived");
            std::thread::yield_now();
        };
        let Output::Preview(img) = got else { panic!("not a preview") };
        assert_eq!(img.rgba, [2, 2, 2, 255]);
        assert!(!w.is_busy());
    }
}

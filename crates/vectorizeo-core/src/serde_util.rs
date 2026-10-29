//! Serde helpers for the stage data (feature `serde`).

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// `&'static str` labels: deserialized strings are interned (each distinct label leaks once).
pub mod static_str {
    use super::*;

    pub fn serialize<S: Serializer>(s: &&'static str, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<&'static str, D::Error> {
        static LABELS: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
        let s = String::deserialize(de)?;
        let mut labels = LABELS.get_or_init(Default::default).lock().expect("label table");
        if let Some(&l) = labels.get(s.as_str()) {
            return Ok(l);
        }
        let l: &'static str = Box::leak(s.into_boxed_str());
        labels.insert(l);
        Ok(l)
    }
}

/// Arrays longer than serde's 32 elements, as sequences.
pub mod big_array {
    use super::*;

    pub fn serialize<S: Serializer, T: Serialize, const N: usize>(a: &[T; N], ser: S) -> Result<S::Ok, S::Error> {
        a.as_slice().serialize(ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>, T: Deserialize<'de>, const N: usize>(
        de: D,
    ) -> Result<[T; N], D::Error> {
        let v = Vec::<T>::deserialize(de)?;
        let len = v.len();
        v.try_into().map_err(|_| serde::de::Error::invalid_length(len, &"the array length"))
    }
}
